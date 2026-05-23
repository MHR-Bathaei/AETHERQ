import subprocess
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. Spawn the optimized C++ Daemon as a background subprocess
# Ensuring stdin/stdout are set to PIPE for seamless IPC streaming
print("[HOST CLIENT] Launching hardware-accelerated AETHER Daemon...")
daemon = subprocess.Popen(
    [r".\aether_daemon.exe"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1 # Line buffered
)

# Consume the daemon's startup handshake message
startup_msg = daemon.stdout.readline().strip()
print(f"[DAEMON HANDSHAKE] {startup_msg}")

# 2. Set up the Matplotlib Dashboard Layout
fig = plt.figure(figsize=(14, 6))
fig.canvas.manager.set_window_title("AETHER Edge AI Telemetry Dashboard (AMD Challenge)")

# Left Subplot: Live 3D Trajectory Vector Space
ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
ax_3d.set_title("Real-Time Physics-Informed GNN Vector Map")
ax_3d.set_xlabel("X Vector")
ax_3d.set_ylabel("Y Vector")
ax_3d.set_zlabel("Z Vector")

# Right Subplot: Live Microsecond Latency Tracker
ax_lat = fig.add_subplot(1, 2, 2)
ax_lat.set_title("Daemon Turnaround Latency (Microseconds)")
ax_lat.set_xlabel("Time Frame")
ax_lat.set_ylabel(r"Latency ($\mu$s)")
ax_lat.grid(True, linestyle="--", alpha=0.6)

# Initialize historical telemetry queues
max_points = 50
history_x, history_y, history_z = [], [], []
latency_history = []
frame_axis = []

# Pre-generate an orbital trajectory loop (A synthetic 3D Helix)
t_space = np.linspace(0, 50, 500)
mock_orbit_x = np.sin(t_space)
mock_orbit_y = np.cos(t_space)
mock_orbit_z = t_space * 0.1

# 3. Dynamic Real-Time Animation Loop
def update(frame):
    global history_x, history_y, history_z, latency_history, frame_axis
    
    # Extract current 3D position state
    pos_x = mock_orbit_x[frame]
    pos_y = mock_orbit_y[frame]
    pos_z = mock_orbit_z[frame]
    
    # Synthesize a 48-element float frame (Mapping node matrices matching GNN specifications)
    # We create variations based on current position to feed the graph structure
    simulated_node_features = []
    for node in range(8):
        simulated_node_features.extend([pos_x * (node*0.1), pos_y, pos_z, 0.1, 0.5, 0.9])
        
    # Convert payload into a single comma-separated string line matching daemon expectations
    payload_string = ",".join([f"{val:.4f}" for val in simulated_node_features]) + "\n"
    
    try:
        # IPC: Stream the telemetry payload directly into the C++ daemon's stdin
        daemon.stdin.write(payload_string)
        daemon.stdin.flush()
        
        # IPC: Instantly read the JSON response string from the daemon's stdout
        response_line = daemon.stdout.readline().strip()
        response_data = json.loads(response_line)
        
        if response_data.get("status") == "success":
            # Extract neural network predictions and hardware profile time metrics
            b_field = response_data["B_field"]
            latency = response_data["latency_us"]
            
            # Append predictions to history queues for tracing paths
            history_x.append(b_field[0])
            history_y.append(b_field[1])
            history_z.append(b_field[2])
            latency_history.append(latency)
            frame_axis.append(frame)
            
            # Keep history limited to prevent memory bloat in visualization window
            if len(history_x) > max_points:
                history_x.pop(0)
                history_y.pop(0)
                history_z.pop(0)
                latency_history.pop(0)
                frame_axis.pop(0)
                
            # Clear plots for redrawing clean frames
            ax_3d.cla()
            ax_3d.set_title("Real-Time Physics-Informed GNN Vector Map")
            ax_3d.plot(history_x, history_y, history_z, color="#00E676", lw=2, label="Predicted B-Field Vector")
            ax_3d.scatter(b_field[0], b_field[1], b_field[2], color="red", s=50, label="Current Core Estimate")
            ax_3d.legend(loc="upper left")
            
            ax_lat.cla()
            ax_lat.set_title("Daemon Turnaround Latency (Microseconds)")
            ax_lat.grid(True, linestyle="--", alpha=0.6)
            ax_lat.plot(frame_axis, latency_history, color="#29B6F6", lw=2)
            ax_lat.set_ylabel(r"Latency ($\mu$s)")
            ax_lat.set_ylim(0, max(max(latency_history) + 10, 50)) # Dynamic bounds with minimum floor
            
    except Exception as e:
        print(f"[ERROR] IPC Loop Failed: {e}")

# Trigger the dynamic display animation running over the trajectory array
ani = FuncAnimation(fig, update, frames=range(500), interval=50, repeat=False)
plt.tight_layout()
plt.show()

# 4. Graceful Cleanup Handshake
print("\n[HOST CLIENT] Window closed. Sending graceful shutdown payload to C++ Daemon...")
try:
    daemon.stdin.write("EXIT\n")
    daemon.stdin.flush()
    daemon.terminate()
    print("[SUCCESS] Hardware service decoupled and closed safely.")
except Exception as e:
    print(f"Cleanup warning: {e}")