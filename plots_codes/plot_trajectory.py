import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm


SCENARIO_CONFIG = {
    2: {
        "user_num": 20,
        "length": 40,
        "width": 40,
        "ini_loc": [14.76, 14.83],
        "end_loc": [27.62, 23.47],
    },
    3: {
        "user_num": 30,
        "length": 40,
        "width": 40,
        "ini_loc": [14.76, 14.83],
        "end_loc": [27.62, 23.47],
    },
    4: {
        "user_num": 40,
        "length": 40,
        "width": 40,
        "ini_loc": [14.76, 14.83],
        "end_loc": [27.62, 23.47],
    },
}

TRAJECTORY_COLORS = [
    "#08306B",
    "#67000D",
    "#3F007D",
    "#7F2704",
    "#00441B",
    "#084594",
    "#49006A",
    "#7A0177",
]


def mkdir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


def get_effective_traj_end(x_series, y_series, t, move_eps=1e-6):
    if t <= 0:
        return 0

    points = np.stack([x_series[: t + 1], y_series[: t + 1]], axis=1)
    move = np.linalg.norm(np.diff(points, axis=0), axis=1)
    moving_idx = np.where(move > move_eps)[0]
    if moving_idx.size == 0:
        return 0
    return int(moving_idx[-1] + 1)


def draw_radio_map(radio_map):
    if radio_map == "None":
        return False

    if radio_map == "A2G":
        npzfile = np.load("results/datas/radiomap/Radio_datas_A2G.npz")
        value = npzfile["arr_1"]
        x_vec = npzfile["arr_2"]
        y_vec = npzfile["arr_3"]
        levels = np.arange(-20, 40, 4)
        cmap = plt.get_cmap("viridis", len(levels) - 1)
        norm = BoundaryNorm(levels, cmap.N)
        plt.contourf(
            np.array(x_vec) * 10,
            np.array(y_vec) * 10,
            value,
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend="both",
        )
        ticks = np.arange(-20, 36, 4)
        cbar = plt.colorbar(ticks=ticks)
        cbar.set_label("SNR", labelpad=20, rotation=270, fontsize=18)
        cbar.ax.tick_params(labelsize=14, width=1.2, length=5)
        cbar.outline.set_linewidth(1.5)
        return True

    if radio_map == "G2A":
        npzfile = np.load("results/datas/radiomap/Radio_datas_G2A.npz")
        value = 1 - npzfile["arr_0"]
        x_vec = npzfile["arr_2"]
        y_vec = npzfile["arr_3"]
        levels = np.linspace(0, 1.0, 11, endpoint=True)
        cmap = plt.get_cmap("viridis", len(levels) - 1)
        norm = BoundaryNorm(levels, cmap.N)
        plt.contourf(
            np.array(x_vec) * 10,
            np.array(y_vec) * 10,
            value,
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend="max",
        )
        cbar = plt.colorbar(ticks=levels)
        cbar.set_label("Coverage Probability", labelpad=20, rotation=270, fontsize=18)
        cbar.ax.tick_params(labelsize=14, width=1.2, length=5)
        cbar.outline.set_linewidth(1.5)
        return True

    raise ValueError(f"Unsupported radio_map: {radio_map}")


def resolve_paths(args):
    scenario = SCENARIO_CONFIG[args.uav_num]
    if args.input:
        input_path = args.input
    elif args.episode is not None:
        input_path = f"results/datas/trajectory/MultiUAV_uav{args.uav_num}_ep{args.episode}.npz"
    else:
        input_path = (
            f"results/test/UAV_{args.uav_num}/{args.model_episode}/"
            f"test_results_uav{args.uav_num}.npz"
        )

    if args.output:
        output_path = args.output
    else:
        base_name = (
            f"trajectory_uav{args.uav_num}_ep{args.episode}.png"
            if args.episode is not None
            else f"trajectory_uav{args.uav_num}_{args.model_episode}.png"
        )
        output_path = os.path.join("results/figs/trajectory", base_name)

    return scenario, input_path, output_path


def load_plot_data(input_path, episode_index=0):
    data = np.load(input_path, allow_pickle=True)

    if "x_uav" in data and "y_uav" in data:
        x_uav = np.asarray(data["x_uav"], dtype=np.float32)
        y_uav = np.asarray(data["y_uav"], dtype=np.float32)
        x_user = np.asarray(data["x_user"], dtype=np.float32)
        y_user = np.asarray(data["y_user"], dtype=np.float32)
        steps = int(data["steps"]) if "steps" in data else int(x_uav.shape[1] - 1)
        completed_targets = int(data["completed_targets"]) if "completed_targets" in data else None
        success = bool(data["success"]) if "success" in data else None
        return {
            "x_uav": x_uav,
            "y_uav": y_uav,
            "x_user": x_user,
            "y_user": y_user,
            "steps": steps,
            "completed_targets": completed_targets,
            "success": success,
        }

    if "x_uav_all" in data and "y_uav_all" in data:
        x_uav_all = np.asarray(data["x_uav_all"], dtype=np.float32)
        y_uav_all = np.asarray(data["y_uav_all"], dtype=np.float32)
        complete_time = np.asarray(data["Complete_time"])
        completed_targets = np.asarray(data["Completed_targets"])

        if episode_index < 0 or episode_index >= x_uav_all.shape[0]:
            raise IndexError(
                f"episode index out of range: {episode_index}, valid 0~{x_uav_all.shape[0] - 1}"
            )

        steps = int(complete_time[episode_index])
        return {
            "x_uav": x_uav_all[episode_index][:, : steps + 1],
            "y_uav": y_uav_all[episode_index][:, : steps + 1],
            "x_user": np.asarray(data["x_user"], dtype=np.float32),
            "y_user": np.asarray(data["y_user"], dtype=np.float32),
            "steps": steps,
            "completed_targets": int(completed_targets[episode_index]),
            "success": None,
        }

    raise KeyError(f"Unsupported trajectory file format: {input_path}")


def draw_trajectory(
    plot_data,
    scenario,
    output_path,
    title=None,
    show_markers=True,
    radio_map="None",
):
    x_uav = plot_data["x_uav"]
    y_uav = plot_data["y_uav"]
    x_user = plot_data["x_user"]
    y_user = plot_data["y_user"]
    steps = int(plot_data["steps"])

    mkdir(os.path.dirname(output_path))

    fig, ax = plt.subplots(figsize=(10, 10), dpi=180)
    has_radio_map = draw_radio_map(radio_map)

    for idx in range(len(x_user)):
        ax.scatter(
            x_user[idx] * 100,
            y_user[idx] * 100,
            c="white" if has_radio_map else "black",
            marker="^",
            s=52,
            label="Inspection Point" if idx == 0 else "",
            edgecolors="black",
            linewidths=0.9,
            zorder=5,
        )

    ax.scatter(
        scenario["ini_loc"][0] * 100,
        scenario["ini_loc"][1] * 100,
        c="red",
        marker="o",
        s=88,
        label="Start",
        edgecolors="black",
        linewidths=1.0,
        zorder=6,
    )
    ax.scatter(
        scenario["end_loc"][0] * 100,
        scenario["end_loc"][1] * 100,
        c="gold",
        marker="s",
        s=88,
        label="End",
        edgecolors="black",
        linewidths=1.0,
        zorder=6,
    )

    for i in range(x_uav.shape[0]):
        color = TRAJECTORY_COLORS[i % len(TRAJECTORY_COLORS)]
        effective_t = get_effective_traj_end(
            x_uav[i], y_uav[i], min(steps, x_uav.shape[1] - 1)
        )
        path_x = x_uav[i][: effective_t + 1] * 100
        path_y = y_uav[i][: effective_t + 1] * 100
        sample_stride = max(1, len(path_x) // 40)

        ax.plot(
            path_x,
            path_y,
            color=color,
            linewidth=2.8,
            alpha=0.96,
            label=f"UAV {i + 1}",
            zorder=4,
        )

        if show_markers:
            ax.scatter(
                path_x[::sample_stride],
                path_y[::sample_stride],
                c=color,
                s=12,
                alpha=0.65,
                zorder=4,
            )
            ax.plot(
                path_x[0],
                path_y[0],
                marker="s",
                markersize=7,
                color=color,
                markeredgecolor="black",
                markeredgewidth=0.8,
                zorder=5,
            )
            ax.plot(
                path_x[-1],
                path_y[-1],
                marker="o",
                markersize=8,
                color=color,
                markeredgecolor="black",
                markeredgewidth=0.8,
                zorder=5,
            )

    ax.set_xlabel("X (m)", fontsize=22)
    ax.set_ylabel("Y (m)", fontsize=22)
    ax.set_xlim((0, scenario["length"] * 100))
    ax.set_ylim((0, scenario["width"] * 100))
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", labelsize=18, width=1.4, length=6, pad=8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    ax.grid(False)

    if has_radio_map:
        legend = ax.legend(
            fontsize=10,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=min(4, x_uav.shape[0] + 3),
            frameon=True,
        )
    else:
        legend = ax.legend(fontsize=10, loc="best", frameon=True)
    legend.get_frame().set_alpha(0.92)

    if title:
        ax.set_title(title, fontsize=16, pad=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Trajectory figure saved to: {output_path}")


def create_parser():
    parser = argparse.ArgumentParser(
        description="Draw UAV trajectories exported by Test_MulUAV.py"
    )
    parser.add_argument("--uav_num", type=int, default=3, choices=sorted(SCENARIO_CONFIG.keys()))
    parser.add_argument("--model_episode", type=str, default="stable")
    parser.add_argument("--episode", type=int, default=None, help="Single episode trajectory id")
    parser.add_argument("--test_episode_index", type=int, default=0, help="Episode index in test_results")
    parser.add_argument("--input", type=str, default=None, help="Explicit input npz file")
    parser.add_argument("--output", type=str, default=None, help="Explicit output image path")
    parser.add_argument("--title", type=str, default="", help="Optional title; empty means no title")
    parser.add_argument("--hide_markers", action="store_true", help="Hide sampled points and endpoint markers")
    parser.add_argument(
        "--radio_map",
        type=str,
        default="None",
        choices=["A2G", "G2A", "None"],
        help="Background radio map type",
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    scenario, input_path, output_path = resolve_paths(args)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Trajectory file not found: {input_path}")

    plot_data = load_plot_data(input_path, episode_index=args.test_episode_index)
    title = None if args.title == "" else args.title

    draw_trajectory(
        plot_data,
        scenario,
        output_path,
        title=title,
        show_markers=not args.hide_markers,
        radio_map=args.radio_map,
    )


if __name__ == "__main__":
    main()
