import numpy as np

def event_detector(pupil_trail, is_blinking):
    """adaptive mark velocity outlier as saccade for eye movement events detection"""
    event_labels = []
    pupil_velocity = np.zeros(pupil_trail.shape)
    pupil_velocity[1:] = pupil_trail[1:] - pupil_trail[:-1]  # diff -> velocity
    pupil_velocity = pupil_velocity
    pupil_acc = np.zeros(pupil_trail.shape)
    pupil_acc[1:] = pupil_velocity[1:] - pupil_velocity[:-1]  # diff -> acc
    pupil_acc = pupil_acc
    N = len(pupil_trail)
    event_labels = ['na'] * N

    # 1. mark velocity outliers(2 std) as saccade
    def mark_true_as(labels, bools, target_label, prlf_len=2):
        N = len(labels)
        for i in range(N):
            if bools[i]:
                labels[i] = target_label
        # if prlf_len > 0:
        #     labels = proliferate_label(labels, target_label, prlf_len)
        return labels

    def proliferate_label(labels, prlf_target, prlf_len):
        N = len(labels)
        to_prlf = [False] * N
        for i in range(N):
            if labels[i] == prlf_target:
                for ii in range(i - prlf_len, i + prlf_len + 1):
                    if 0 <= ii <= N - 1:
                        to_prlf[ii] = True
        labels = np.array(labels)
        labels[to_prlf] = prlf_target
        return list(labels)
        return labels

    def calculate_euclidean_dist(arr):
        return np.sqrt(np.sum((arr ** 2), axis=1))

    def generate_outlier_mark(arr, label, outlier_label=' ', num_std=1):
        not_outlier_arr = []
        for i in range(len(arr)):
            if label[i] != outlier_label:
                not_outlier_arr.append(arr[i])
        not_outlier_arr = np.array(not_outlier_arr)
        m, std = np.mean(not_outlier_arr), np.std(not_outlier_arr)
        outlier_threshold = num_std * std + m
        marks_keep = arr < outlier_threshold
        for i in range(len(arr)):
            if label[i] == outlier_label:
                marks_keep[i] = False
        return marks_keep, outlier_threshold

    pupil_velocity_abs = calculate_euclidean_dist(pupil_velocity)
    num_std = 2


    # DriveGaze 3
    # EVE 2
    # Gaze360 2

    # Savitzky–Golay filter
    # EVE/DriveGaze 4
    # Gaze360 4
    # # print(len(pupil_velocity_abs))
    # # pupil_velocity_abs_filtered = scipy.signal.savgol_filter(pupil_velocity_abs, window_length=4, polyorder=2)

    pupil_velocity_abs_filtered = pupil_velocity_abs

    is_not_jumping, threshold_old = generate_outlier_mark(pupil_velocity_abs_filtered, label=event_labels,
                                                          outlier_label='saccade', num_std=num_std)
    is_jumping = ~is_not_jumping
    event_labels = mark_true_as(event_labels, is_jumping, 'saccade', prlf_len=1)  # safe margin of 2 frames (32ms)
    # adaptive algorithm
    while 1:
        is_not_jumping, threshold = generate_outlier_mark(pupil_velocity_abs_filtered, label=event_labels,
                                                          outlier_label='saccade', num_std=num_std)
        # print(f'threshold_old:{threshold_old},threshold_now:{threshold}')
        is_jumping = ~is_not_jumping
        event_labels = mark_true_as(event_labels, is_jumping, 'saccade', prlf_len=1)  # safe margin of 2 frames (32ms)

        # # Draw velocity and acc

        # for l in set(event_labels):
        #     x_d = [i for i in range(int(len(y)/100)) if event_labels[i] == l]
        #     y_d = [1 for i in range(int(len(y)/100)) if event_labels[i] == l]
        #     plt.plot(x_d, y_d, 'o', label = l)

        # fig = plt.gcf()
        # fig.set_size_inches(18.5, 10.5)
        # plt.xlabel('Frame')
        # plt.legend()
        # dst_filename = './eye_movement_events_classify_result/label_visualize.jpg'
        # plt.savefig(dst_filename)
        # plt.clf()
        # input()

        if abs(threshold - threshold_old) <= 1:
            print('break')
            break
        threshold_old = threshold

    # 2. mark blink by blink_bools
    event_labels = mark_true_as(event_labels, is_blinking, 'blink', prlf_len=1)  # also add a 2 frame margin

    for i in range(len(event_labels)):
        if event_labels[i] != 'saccade' and event_labels[i] != 'blink':
            event_labels[i] = 'fixation'
    # return event_labels, pupil_velocity_abs, pupil_velocity_abs_filtered, threshold_old
    return event_labels