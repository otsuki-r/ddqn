import matplotlib.pyplot as plt
import pathlib
from matplotlib import animation


def save_frames_as_gif(frames, path: pathlib.Path) -> None:
    """
    Collates frames saved from a run into a GIF.

    Parameters
    ----------
    frames : list[]
        Frames to collect.
    path : pathlib.Path
        Path to save output GIF to.

    Notes
    -----
    Adapted from
    https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553
    """

    # Mess with this to change frame size
    plt.figure(
        figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72
    )

    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames=len(frames), interval=50
    )
    anim.save(str(path), writer="imagemagick", fps=60)
