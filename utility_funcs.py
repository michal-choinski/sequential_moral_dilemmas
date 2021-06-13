import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


def rgb2gray(rgb):
    return np.mean(rgb, axis=2).reshape(rgb.shape[0], rgb.shape[1], 1)


def save_img(rgb_arr, path, name):
    plt.imshow(rgb_arr, interpolation='nearest')
    plt.savefig(path + name)


def make_video_from_image_dir(vid_path, img_folder, video_name='trajectory', fps=5):
    """
    Create a video from a directory of images
    """
    images = [img for img in os.listdir(img_folder) if img.endswith(".png")]
    images.sort()

    rgb_imgs = []
    for i, image in enumerate(images):
        img = cv2.imread(os.path.join(img_folder, image))
        rgb_imgs.append(img)

    make_video_from_rgb_imgs(rgb_imgs, vid_path, video_name=video_name, fps=fps)


def make_video_from_rgb_imgs(rgb_arrs, vid_path, video_name='trajectory',
                             fps=5, format="mp4v", resize=(640, 480)):
    """
    Create a video from a list of rgb arrays
    """
    print("Rendering video...")
    if vid_path[-1] != '/':
        vid_path += '/'
    video_path = vid_path + video_name + '.mp4'

    if resize is not None:
        width, height = resize
    else:
        frame = rgb_arrs[0]
        height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*format)
    video = cv2.VideoWriter(video_path, fourcc, float(fps), (width, height))

    for i, image in enumerate(rgb_arrs):
        percent_done = int((i / len(rgb_arrs)) * 100)
        if percent_done % 20 == 0:
            print("\t...", percent_done, "% of frames rendered")
        if resize is not None:
            image = cv2.resize(image, resize, interpolation=cv2.INTER_NEAREST)
        video.write(image)

    video.release()
    #cv2.destroyAllWindows()


def make_video_from_rgb_imgs_and_figs(rgb_arrs, figs, vid_path, video_name='trajectory_with_figures',
                             fps=20, format="mp4v", resize=(2720, 1440)):
    """
    Create a video from a list of rgb arrays
    """
    print("Rendering video...")
    if vid_path[-1] != '/':
        vid_path += '/'
    video_path = vid_path + video_name + '.mp4'

    if resize is not None:
        width, height = resize
    else:
        frame = rgb_arrs[0]
        height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*format)
    video = cv2.VideoWriter(video_path, fourcc, float(fps), (width, height))

    for i, image in enumerate(rgb_arrs):
        percent_done = int((i / len(rgb_arrs)) * 100)
        if percent_done % 20 == 0:
            print("\t...", percent_done, "% of frames rendered")

        image = cv2.resize(image, (1440, 1440), interpolation=cv2.INTER_NEAREST)
        image_rewards = cv2.resize(figs[i]["rewards"], (640, 480), interpolation=cv2.INTER_NEAREST)
        image_gathering = cv2.resize(figs[i]["gathering"], (640, 480), interpolation=cv2.INTER_NEAREST)
        image_apples = cv2.resize(figs[i]["apples"], (640, 480), interpolation=cv2.INTER_NEAREST)
        image_starvation = cv2.resize(figs[i]["starvation"], (640, 480), interpolation=cv2.INTER_NEAREST)
        image_aid = cv2.resize(figs[i]["aid"], (640, 480), interpolation=cv2.INTER_NEAREST)
        image_contribution = cv2.resize(figs[i]["contribution"], (640, 480), interpolation=cv2.INTER_NEAREST)

        image_figures_col1 = cv2.vconcat([image_rewards, image_apples, image_aid])
        image_figures_col2 = cv2.vconcat([image_starvation, image_gathering, image_contribution])
        image = cv2.hconcat([image, image_figures_col1, image_figures_col2])
        video.write(image)

    video.release()
    #cv2.destroyAllWindows()


def return_view(grid, pos, row_size, col_size):
    """Given a map grid, position and view window, returns correct map part

    Note, if the agent asks for a view that exceeds the map bounds,
    it is padded with zeros

    Parameters
    ----------
    grid: 2D array
        map array containing characters representing
    pos: list
        list consisting of row and column at which to search
    row_size: int
        how far the view should look in the row dimension
    col_size: int
        how far the view should look in the col dimension

    Returns
    -------
    view: (np.ndarray) - a slice of the map for the agent to see
    """
    x, y = pos
    left_edge = x - col_size
    right_edge = x + col_size
    top_edge = y - row_size
    bot_edge = y + row_size
    pad_mat, left_pad, top_pad = pad_if_needed(left_edge, right_edge,
                                               top_edge, bot_edge, grid)
    x += left_pad
    y += top_pad
    view = pad_mat[x - col_size: x + col_size + 1,
                   y - row_size: y + row_size + 1]
    return view


def pad_if_needed(left_edge, right_edge, top_edge, bot_edge, matrix):
    # FIXME(ev) something is broken here, I think x and y are flipped
    row_dim = matrix.shape[0]
    col_dim = matrix.shape[1]
    left_pad, right_pad, top_pad, bot_pad = 0, 0, 0, 0
    if left_edge < 0:
        left_pad = abs(left_edge)
    if right_edge > row_dim - 1:
        right_pad = right_edge - (row_dim - 1)
    if top_edge < 0:
        top_pad = abs(top_edge)
    if bot_edge > col_dim - 1:
        bot_pad = bot_edge - (col_dim - 1)

    return pad_matrix(left_pad, right_pad, top_pad, bot_pad, matrix, 0), left_pad, top_pad


def pad_matrix(left_pad, right_pad, top_pad, bot_pad, matrix, const_val=1):
    pad_mat = np.pad(matrix, ((left_pad, right_pad), (top_pad, bot_pad)),
                     'constant', constant_values=(const_val, const_val))
    return pad_mat

