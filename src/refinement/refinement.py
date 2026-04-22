import numpy as np


def refine(PoG_x_pred, PoG_y_pred, history_PoG, history_blink, w_screen, h_screen):

    #refinement_Validaty
    history_PoG = [value for idx, value in enumerate(history_PoG) if not history_blink[idx]]

    range_scale = 0.9
    history_percentile = np.percentile(history_PoG,
                                       [100 * (1 - range_scale) / 2, 100 - 100 * (1 - range_scale) / 2],
                                       axis=0)  # get percentile
    # print(pred_percentile)
    history_percentile_range = history_percentile[1] - history_percentile[0]

    '''
    # refinement_Zoom
    truth_percentile_range = [w_screen, h_screen] * range_scale
    zoom_scale = truth_percentile_range / history_percentile_range'''

    # refinement_SC
    gtr = [w_screen / 2, h_screen / 2]
    # SC_history_average = np.mean(np.array(history_PoG), axis=0)
    SC_history_average = np.mean(history_percentile, axis=0)
    offset = gtr - SC_history_average
    # print(offset)
    PoG_x_pred = PoG_x_pred + offset[0]
    PoG_y_pred = PoG_y_pred + offset[1]

    return PoG_x_pred, PoG_y_pred

import numpy as np


def refine_x(PoG_x_pred, PoG_y_pred, history_PoG, history_blink, w_screen, h_screen):

    #refinement_Validaty
    history_PoG = [value for idx, value in enumerate(history_PoG) if not history_blink[idx]]

    range_scale = 0.9
    history_percentile = np.percentile(history_PoG,
                                       [100 * (1 - range_scale) / 2, 100 - 100 * (1 - range_scale) / 2],
                                       axis=0)  # get percentile

    history_percentile_range = history_percentile[1] - history_percentile[0]
    # print("history_percentile: \n", history_percentile)

    # refinement_Zoom
    truth_percentile_range = [k * range_scale for k in [w_screen, h_screen]]
    zoom_scale = truth_percentile_range / history_percentile_range
    # print("zoom_scale: \n", zoom_scale)

    # refinement_SC
    gtr = [w_screen / 2, h_screen / 2]
    SC_history_average = np.mean(history_percentile, axis=0)
    offset = gtr - SC_history_average

    # 实现平移和缩放
    PoG_x_pred = PoG_x_pred + offset[0]
    # PoG_x_pred = (PoG_x_pred - w_screen / 2) * zoom_scale[0] + w_screen / 2

    return PoG_x_pred, PoG_y_pred, offset[0]