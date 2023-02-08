import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import plotly.express as px
import numpy as np
import streamlit as st
import torchvision.transforms as transforms

from torchmetrics import JaccardIndex
from pathlib import Path
from EvaluationScript import evaluate, get_last_modified_model_time_stamp, get_last_modified_epoch, \
    calculate_iou_distribution


def main():
    st.title('Threshold selection tool')

    if 'results' not in st.session_state:
        st.session_state.results = []
    threshold = st.select_slider('Select masking threshold', np.arange(0.0, 1.0, 0.01))

    # Model selection
    checkpoint_path = Path('./checkpoints')
    time_stamp_selection = sorted([x.stem for x in checkpoint_path.glob('*')], reverse=True)
    if len(time_stamp_selection) == 0:
        return
    last_modified_timestamp = get_last_modified_model_time_stamp(checkpoint_path)
    last_modified_idx = [i for i, x in enumerate(time_stamp_selection) if x == last_modified_timestamp][0]
    st.selectbox('Input timestamp', options=time_stamp_selection, index=last_modified_idx, key='timestamp')
    epoch_selection = sorted([x.stem for x in (checkpoint_path / st.session_state.timestamp).glob('*')], reverse=True)
    last_modified_epoch = get_last_modified_epoch(checkpoint_path, st.session_state.timestamp)
    last_modified_epoch_idx = [i for i, x in enumerate(epoch_selection) if x == last_modified_epoch][0]
    st.selectbox('Input epoch', options=epoch_selection, index=last_modified_epoch_idx, key='epoch')

    # Refresh all predictions
    if st.button('Refresh'):
        st.session_state.results = evaluate(st.session_state.timestamp, st.session_state.epoch)
        st.session_state.results = sorted(st.session_state.results, key=lambda x: x.path.stem)

    if st.button('Show threshold/IOU distribution'):
        st.header('Threshold/IOU distribution')
        if len(st.session_state.results) > 0:
            thresholds, ious = calculate_iou_distribution(st.session_state.results)
            plot = px.scatter(x=thresholds, y=ious, labels={'x': 'Threshold', 'y': 'IOU'})
            st.write(f'Maximum IOU {np.max(ious)} for threshold {thresholds[np.argmax(ious)]}')
            st.plotly_chart(plot)


    # Show masked results
    st.header('Results')
    iou_fnc = JaccardIndex(task='binary', threshold=threshold)
    for result in st.session_state.results:
        iou = np.nan
        if result.ground_truth_mask_tensor is not None:
            iou = iou_fnc(result.predicted_mask, result.ground_truth_mask_tensor)
        with st.expander(str(result.path)):
            st.write(f'IOU: {iou}')
            c1, c2, c3 = st.columns(3)

            # Show image, ground truth mask and predicted mask
            with c1:
                st.image(result.image_pil)
            if result.ground_truth_mask_pil is not None:
                with c2:
                    st.image(result.ground_truth_mask_pil)
            with c3:
                mask = transforms.ToPILImage()((result.predicted_mask > threshold).float())
                st.image(mask)




if __name__ == '__main__':
    main()
