import os
import subprocess
import time
import textwrap

def run_cmd(cmd, stream=True):
    """Runs a shell command. Streams output if requested."""
    print(f"\n[EXEC] {' '.join(cmd)}")
    if stream:
        subprocess.run(cmd, check=True)
    else:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()

def main():
    print("=== Kubernetes Pod Deployer & Executor ===\n")
    
    # 1. Interactive Inputs
    code_dir = "inca_cuda_run".strip()
    registry_prefix = "nishant556/".strip()#input("Enter your Docker registry prefix (e.g., username/ or registry.xyz.com/): ").strip()
    app_name = "gpu-worker".strip()#input("Enter application/image name (e.g., gpu-worker): ".strip()
    image_tag = "v1".strip()#input("Enter image tag (e.g., v1): ").strip()
    base_image = "python:3.10-slim".strip()#input("Enter base Docker image (e.g., python:3.10-slim or pytorch/pytorch:latest-cuda): ").strip()
    entry_cmd = "python run_cuda".strip()#input("Enter the command to execute your code inside the pod (e.g., python main.py): ").strip()
    
    full_image_name = f"{registry_prefix}{app_name}:{image_tag}"
    pod_name = f"{app_name}-pod"
    
    # 2. Generate Dockerfile
    print("\n--- Generating Dockerfile ---")
    dockerfile_content = textwrap.dedent(f"""\
        FROM {base_image}
        WORKDIR /app
        COPY {code_dir} /app
        # Add pip install requirements.txt here if needed
        CMD ["sleep", "infinity"] 
    """)
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    print("Dockerfile created.")

    # 3. Build and Push Docker Image
    print("\n--- Building Docker Image ---")
    run_cmd(["docker", "build", "-t", full_image_name, "-f", "Dockerfile", "."])
    
    print("\n--- Pushing Docker Image to Registry ---")
    # Pushing is required so your remote Kubernetes cluster can pull it
    run_cmd(["docker", "push", full_image_name])

    # 4. Generate Kubernetes YAML (with GPU support)
    print("\n--- Generating Pod YAML ---")
    yaml_content = textwrap.dedent(f"""\
        apiVersion: v1
        kind: Pod
        metadata:
          name: {pod_name}
          labels:
            app: {app_name}
        spec:
          containers:
          - name: {app_name}-container
            image: {full_image_name}
            imagePullPolicy: Always
            command: ["/bin/sh", "-c", "sleep infinity"]
            resources:
              limits:
                nvidia.com/gpu: 1 # Requests 1 GPU
          restartPolicy: Never
    """)
    
    yaml_file = f"{pod_name}.yaml"
    with open(yaml_file, "w") as f:
        f.write(yaml_content)
    print(f"{yaml_file} created.")

    
    # 5. Deploy to Kubernetes via SSH
    print("\n--- Deploying to Remote Kubernetes ---")
    # Delete existing pod on remote server
    run_cmd(["ssh", ssh_target, "kubectl", "delete", "pod", pod_name, "--ignore-not-found=true"])
    
    # Pass the generated YAML string directly into the remote kubectl command via stdin
    print(f"\n[EXEC] ssh {ssh_target} kubectl apply -f -")
    subprocess.run(["ssh", ssh_target, "kubectl", "apply", "-f", "-"], input=yaml_content.encode('utf-8'), check=True)

    # 6. Wait for Pod to be Ready (Polling via SSH)
    print(f"\n--- Waiting for {pod_name} to be Running ---")
    while True:
        status = run_cmd(["ssh", ssh_target, "kubectl", "get", "pod", pod_name, "-o", "jsonpath={.status.phase}"], stream=False)
        if status == "Running":
            print(f"Pod {pod_name} is now Running!")
            break
        elif status in ["Failed", "Error"]:
            print(f"Pod failed to start. Status: {status}")
            return
        print(f"Current status: {status}... waiting 3 seconds.")
        time.sleep(3)

    # 7. Execute Code Inside the Pod via SSH
    print("\n--- Executing Code Inside Pod ---")
    # We use -t with SSH to force a pseudo-terminal allocation, which is required for kubectl exec -it to work remotely
    exec_cmd = ["ssh", "-t", ssh_target, "kubectl", "exec", "-it", pod_name, "--", "/bin/sh", "-c", entry_cmd]
    run_cmd(exec_cmd)
    
    print("\n=== Workflow Complete ===")

if __name__ == "__main__":
    main()