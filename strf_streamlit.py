import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import pandas as pd
import random

st.set_page_config(layout="wide")
st.title("STRF Scheduling (without quantum time)")

# User inputs
num_jobs = st.number_input("Number of Jobs", 1, 10, 3)
num_cpus = st.number_input("Number of CPUs", 1, 4, 2)
chunk_unit = st.number_input("Chunk Unit (e.g., 0.5, 1.0):", value=1.0)

# Randomize job times
if st.button("Randomize Job Times"):
    st.session_state.special_jobs = [
        {"arrival": round(random.uniform(0, 5) * 2) / 2, "burst": round(random.uniform(1, 10) * 2) / 2}
        for _ in range(num_jobs)
    ]

# Input job times
processes = []
for i in range(num_jobs):
    st.subheader(f"Job J{i+1}")
    default_arrival = st.session_state.get("special_jobs", [{}]*num_jobs)[i].get("arrival", 0.0)
    default_burst = st.session_state.get("special_jobs", [{}]*num_jobs)[i].get("burst", 3.0)

    arrival = st.number_input(f"Arrival Time for J{i+1}", value=default_arrival, key=f"a_{i}")
    burst = st.number_input(f"Burst Time for J{i+1}", value=default_burst, key=f"b_{i}")
    processes.append({'id': f'J{i+1}', 'arrival_time': arrival, 'burst_time': burst})

# Run simulation
if st.button("Run Special STRF"):
    arrival_time = {p['id']: p['arrival_time'] for p in processes}
    burst_time = {p['id']: p['burst_time'] for p in processes}
    remaining_time = burst_time.copy()
    start_time, end_time, job_chunks = {}, {}, {}
    gantt_data, queue_snapshots = [], []
    busy_jobs = set()
    current_time = 0
    jobs_completed = 0

    for job_id, total in burst_time.items():
        chunks, remaining = [], total
        while remaining > 0:
            chunk = min(chunk_unit, remaining)
            chunks.append(chunk)
            remaining -= chunk
        job_chunks[job_id] = chunks

    cpu_names = [f"CPU{i+1}" for i in range(num_cpus)]
    busy_until = {cpu: 0 for cpu in cpu_names}
    current_jobs = {cpu: None for cpu in cpu_names}

    def capture_queue(time, available_jobs):
        queue = sorted([j for j in available_jobs if remaining_time[j] > 0],
                       key=lambda j: (remaining_time[j], arrival_time[j]))
        if queue:
            job_info = [(j, round(remaining_time[j], 1)) for j in queue]
            queue_snapshots.append((time, job_info))

    initial_jobs = [p['id'] for p in processes if p['arrival_time'] <= current_time]
    capture_queue(current_time, initial_jobs)

    while jobs_completed < num_jobs:
        for cpu in cpu_names:
            if busy_until[cpu] <= current_time and current_jobs[cpu] is not None:
                job_id = current_jobs[cpu]
                busy_jobs.discard(job_id)
                current_jobs[cpu] = None

        available_cpus = [cpu for cpu in cpu_names if busy_until[cpu] <= current_time and current_jobs[cpu] is None]
        available_jobs = [j for j in remaining_time if remaining_time[j] > 0 and arrival_time[j] <= current_time and j not in busy_jobs]

        if available_cpus and available_jobs:
            capture_queue(current_time, available_jobs)

            available_jobs.sort(key=lambda j: (remaining_time[j], arrival_time[j]))

            for cpu in available_cpus:
                if not available_jobs:
                    break
                job_id = available_jobs.pop(0)
                chunk = job_chunks[job_id].pop(0)
                if job_id not in start_time:
                    start_time[job_id] = current_time

                busy_jobs.add(job_id)
                current_jobs[cpu] = job_id
                remaining_time[job_id] -= chunk
                busy_until[cpu] = current_time + chunk
                gantt_data.append((current_time, cpu, job_id, chunk))

                if remaining_time[job_id] < 1e-3:
                    end_time[job_id] = current_time + chunk
                    jobs_completed += 1

        future_times = (
            [busy_until[c] for c in cpu_names if busy_until[c] > current_time] +
            [arrival_time[j] for j in arrival_time if arrival_time[j] > current_time and remaining_time[j] > 0]
        )
        current_time = min(future_times) if future_times else current_time + 0.1

    # Results
    for p in processes:
        p['start_time'] = start_time[p['id']]
        p['end_time'] = end_time[p['id']]
        p['turnaround_time'] = p['end_time'] - p['arrival_time']

    df = pd.DataFrame([{
        "Job": p['id'],
        "Arrival": p['arrival_time'],
        "Burst": p['burst_time'],
        "Start": round(p['start_time'], 1),
        "End": round(p['end_time'], 1),
        "Turnaround": round(p['turnaround_time'], 1)
    } for p in processes])

    avg_turnaround = sum(p['turnaround_time'] for p in processes) / num_jobs

    st.subheader("Result Table")
    st.dataframe(df, use_container_width=True)
    st.write(f"**Average Turnaround Time:** `{avg_turnaround:.2f}`")

    # Gantt chart
    def plot_gantt():
        fig, ax = plt.subplots(figsize=(18, 8))
        cmap = plt.colormaps['tab20']
        colors = {f'J{i+1}': mcolors.to_hex(cmap(i / max(num_jobs, 1))) for i in range(num_jobs)}
        y_pos = {cpu: num_cpus - idx for idx, cpu in enumerate(cpu_names)}

        for start, cpu, job, dur in gantt_data:
            ax.barh(y=y_pos[cpu], width=dur, left=start, color=colors[job], edgecolor='black')
            ax.text(start + dur / 2, y_pos[cpu], job, ha='center', va='center', color='white')

        for t in range(int(max(end_time.values())) + 1):
            ax.axvline(x=t, color='gray', linestyle='--', linewidth=0.5)

        for t, queue in queue_snapshots:
            for i, (jid, rem) in enumerate(queue):
                rect_y = -1 - i * 0.6
                rect = patches.Rectangle((t - 0.25, rect_y - 0.25), 0.5, 0.5, edgecolor='black', facecolor='white')
                ax.add_patch(rect)
                ax.text(t, rect_y, f"{jid}={rem}", ha='center', va='center', fontsize=7)

        if queue_snapshots:
            max_len = max(len(q[1]) for q in queue_snapshots)
            ax.set_ylim(-1 - max_len * 0.6 - 1, num_cpus + 1)

        ax.set_yticks(list(y_pos.values()))
        ax.set_yticklabels(cpu_names)
        ax.set_xlabel("Time")
        ax.set_title("Gantt Chart - Special STRF")
        plt.grid(axis='x')
        return fig

    st.subheader("Gantt Chart")
    st.pyplot(plot_gantt(), use_container_width=True)
