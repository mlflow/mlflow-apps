

def get_feature_cols(feat_cols, label_col, all_cols):
    # This is the case if the user specified which columns are to be feature columns.
    if feat_cols:
        return feat_cols
    # If no feature columns are specified, it is assumed all columns but the label are features.
    else:
        feats = []
        for col in all_cols:
            if col != label_col:
                feats.append(col)
        return feats
