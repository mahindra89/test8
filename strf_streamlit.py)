import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import pandas as pd
import random

st.set_page_config(layout="wide")
st.title("STRF Scheduling Simulator (Normal Python Version to Streamlit)")

# Inputs
num_jobs = st.number_input("Number of Jobs:", min_value=1, max_value=10, value=3)
num_cpus = st.number_input("Number of CPUs:", min_value=1, max_value=4, value=2)
chunk_unit = st.number_input("Chunk Time Unit (e.g., 0.5, 1.0, 2.0):", value=1.0)
quantum_time = st.number_input("Quantum Time Interval:", value=2.0)

# Optional random generator
if st.button("Randomize Jobs"):
    st.session_state.random_jobs = [
        {"arrival": round(random.uniform(0, 5) * 2) / 2, "burst": round(random.uniform(1, 10) * 2) / 2}
        for _ in range(num_jobs)
    ]

# Input for each job
processes = []
for i in range(num_jobs):
    st.subheader(f"Job J{i+1}")
    default_arrival = st.session_state.get("random_jobs", [{}] * num_jobs)[i].get("arrival", 0.0)
    default_burst = st.session_state.get("random_jobs", [{}] * num_jobs)[i].get("burst", 3.0)

    arrival = st.number_input(f"Arrival Time (J{i+1})", value=default_arrival, key=f"arrival_{i}")
    burst = st.number_input(f"Burst Time (J{i+1})", value=default_burst, key=f"burst_{i}")
    processes.append({'id': f'J{i+1}', 'arrival_time': arrival, 'burst_time': burst})

if st.button("Run STRF Simulation"):
    # Setup state
    arrival_time = {p['id']: p['arrival_time'] for p in processes}
    burst_time = {p['id']: p['burst_time'] for p in processes}
    remaining_time = {p['id']: p['burst_time'] for p in processes}
    start_time = {}
    end_time = {}
    job_chunks = {}

    for job_id, total_time in burst_time.items():
        chunks = []
        remaining = total_time
        while remaining > 0:
            chunk = min(chunk_unit, remaining)
            chunks.append(chunk)
            remaining -= chunk
        job_chunks[job_id] = chunks

    cpu_names = [f"CPU{i+1}" for i in range(num_cpus)]
    busy_until = {cpu: 0 for cpu in cpu_names}
    current_jobs = {cpu: None for cpu in cpu_names}
    busy_jobs = set()
    gantt_data = []
    queue_snapshots = []
    current_time = 0
    jobs_completed = 0
    next_scheduling_time = 0

    def capture_queue_state(time, available_jobs):
        active_jobs = [j for j in available_jobs if remaining_time[j] > 0]
        queue = sorted(active_jobs, key=lambda job_id: (remaining_time[job_id], arrival_time[job_id]))
        job_info = [(job, round(remaining_time[job], 1)) for job in queue]
        if job_info:
            queue_snapshots.append((time, job_info))

    initial_available_jobs = [p['id'] for p in processes if p['arrival_time'] <= current_time]
    capture_queue_state(current_time, initial_available_jobs)

    while jobs_completed < len(processes):
        for cpu, busy_time in list(busy_until.items()):
            if busy_time <= current_time and current_jobs[cpu] is not None:
                job_id = current_jobs[cpu]
                if job_id in busy_jobs:
                    busy_jobs.remove(job_id)
                current_jobs[cpu] = None

        can_schedule = current_time >= next_scheduling_time
        available_cpus = [cpu for cpu in cpu_names if busy_until[cpu] <= current_time and current_jobs[cpu] is None]
        available_jobs = [job_id for job_id in remaining_time
                          if remaining_time[job_id] > 0 and arrival_time[job_id] <= current_time and job_id not in busy_jobs]

        if can_schedule and available_cpus and available_jobs:
            capture_queue_state(current_time, available_jobs)
            available_jobs.sort(key=lambda job_id: (remaining_time[job_id], arrival_time[job_id]))

            for cpu in available_cpus:
                if not available_jobs:
                    break
                selected_job = available_jobs.pop(0)
                if selected_job not in start_time:
                    start_time[selected_job] = current_time

                chunk_size = job_chunks[selected_job].pop(0)
                busy_jobs.add(selected_job)
                current_jobs[cpu] = selected_job
                remaining_time[selected_job] -= chunk_size
                busy_until[cpu] = current_time + chunk_size
                gantt_data.append((current_time, cpu, selected_job, chunk_size))

                if abs(remaining_time[selected_job]) < 0.001:
                    end_time[selected_job] = current_time + chunk_size
                    jobs_completed += 1

            next_scheduling_time = current_time + quantum_time

        next_time_events = []
        next_time_events.extend([busy_until[cpu] for cpu in busy_until if busy_until[cpu] > current_time])
        next_time_events.extend([arrival_time[j] for j in arrival_time if arrival_time[j] > current_time and remaining_time[j] > 0])
        if next_scheduling_time > current_time:
            next_time_events.append(next_scheduling_time)

        if next_time_events:
            current_time = min(next_time_events)
        else:
            break

    for p in processes:
        p['start_time'] = start_time[p['id']]
        p['end_time'] = end_time[p['id']]
        p['turnaround_time'] = p['end_time'] - p['arrival_time']

    avg_turnaround = sum(p['turnaround_time'] for p in processes) / len(processes)

    result_df = pd.DataFrame([{
        'Job': p['id'],
        'Arrival': p['arrival_time'],
        'Burst': p['burst_time'],
        'Start': round(p['start_time'], 1),
        'End': round(p['end_time'], 1),
        'Turnaround': round(p['turnaround_time'], 1)
    } for p in processes])

    st.subheader("Job Summary Table")
    st.dataframe(result_df, use_container_width=True)
    st.write(f"**Average Turnaround Time:** `{avg_turnaround:.2f}`")

    # Gantt Chart Drawing
    def draw_gantt_with_queue():
        max_time = max(end_time.values())
        fig, ax = plt.subplots(figsize=(18, 8))
        cmap = plt.colormaps.get_cmap('tab20')
        colors = {f'J{i+1}': mcolors.to_hex(cmap(i / max(len(processes), 1))) for i in range(len(processes))}
        cpu_ypos = {cpu: num_cpus - idx for idx, cpu in enumerate(cpu_names)}

        for start_time, cpu, job, duration in gantt_data:
            y_pos = cpu_ypos[cpu]
            ax.barh(y=y_pos, width=duration, left=start_time,
                    color=colors[job], edgecolor='black')
            ax.text(start_time + duration / 2, y_pos, job,
                    ha='center', va='center', color='white', fontsize=9)

        for t in range(0, int(max_time) + 1, int(quantum_time)):
            ax.axvline(x=t, color='red', linestyle='-', alpha=0.5, linewidth=0.5)
        for t in range(int(max_time) + 1):
            if t % int(quantum_time) != 0:
                ax.axvline(x=t, color='black', linestyle='--', alpha=0.3)

        ax.set_yticks(list(cpu_ypos.values()))
        ax.set_yticklabels(cpu_ypos.keys())
        ax.set_xlim(0, max_time + 0.5)
        ax.set_xlabel("Time (seconds)")
        ax.set_title(f"Multi-CPU STRF with Chunks = {chunk_unit}s & Quantum Time = {quantum_time}s")

        # Queue snapshot display
        queue_y_base = -1
        for time, job_queue in queue_snapshots:
            for i, (job_id, remaining) in enumerate(job_queue):
                box_y = queue_y_base - i * 0.6
                rect = patches.Rectangle((time - 0.25, box_y - 0.25), 0.5, 0.5,
                                         linewidth=1, edgecolor='black', facecolor='white', fill=True)
                ax.add_patch(rect)
                ax.text(time, box_y, f"{job_id}={remaining}", ha='center', va='center', fontsize=7)

        if queue_snapshots:
            max_len = max(len(q[1]) for q in queue_snapshots)
            min_y = queue_y_base - max_len * 0.6 - 0.5
            ax.set_ylim(min_y, max(cpu_ypos.values()) + 1)

        ax.plot([], [], color='red', linestyle='-', alpha=0.5, label='Quantum Time')
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.grid(axis='x')
        return fig

    # Show the chart
    st.subheader("Gantt Chart (with Queue Snapshots)")
    st.pyplot(draw_gantt_with_queue(), use_container_width=True)
