def definde_NWCSAF_variables(missing_labels=None):
    ct_colors = [
        "#007800",
        "#000000",
        "#fabefa",
        "#dca0dc",
        "#ff6400",
        "#ffb400",
        "#f0f000",
        "#d7d796",
        "#e6e6e6",
        "#c800c8",
        "#0050d7",
        "#00b4e6",
        "#00f0f0",
        "#5ac8a0",
    ]

    ct_indices = [
        1.5,
        2.5,
        3.5,
        4.5,
        5.5,
        6.5,
        7.5,
        8.5,
        9.5,
        10.5,
        11.5,
        12.5,
        13.5,
        14.5,
        15.5,
    ]

    ct_labels = [
        "land",
        "sea",
        "snow",
        "sea ice",
        "very low",
        "low",
        "middle",
        "high opaque",
        "very high opaque",
        "fractional",
        "semi. thin",
        "semi. mod. thick",
        "semi. thick",
        "semi. above low",
        "semi. above snow",
    ]

    if missing_labels is not None:
        mis_ind = [ct_labels.index(ml) for ml in missing_labels]
        for ind in sorted(mis_ind, reverse=True):
            for ct_list in [ct_colors, ct_labels, ct_indices]:
                del ct_list[ind]

    return ct_colors, ct_indices, ct_labels


def check_nwcsaf_version(labels, verbose):
    """
    checks if cloud type labels are mapped by the 2013-netcdf standard

    Uses layers 16-19 (only present at 2013 standard)
    and 7,9,11,13 (only present at 2018 standard)

    Parameters
    ----------
    labels : array like
        Array of labels

    Returns
    -------
    string or None
        String naming the used version or None if version couldnt be determined
    """
    high_sum = odd_sum = 0
    for i in range(16, 20):
        high_sum += (labels == i).sum()
    for i in range(7, 14, 2):
        odd_sum = (labels == i).sum()

    if high_sum > 0 and odd_sum == 0:
        return "v2013"
    if high_sum == 0 and odd_sum > 0:
        return "v2018"
    return None


def switch_nwcsaf_version(labels, target_version, input_version=None):
    """
    maps netcdf cloud types from the 2013 standard to the 2018 standard
    """
    if input_version is None:
        input_version = check_nwcsaf_version(labels)
    if target_version == input_version:
        return labels
    if target_version == "v2018":
        return switch_2018(labels)
    if target_version == "v2013":
        return switch_2013(labels)


def switch_2018(labels):
    """
    maps netcdf cloud types from the 2013 standard to the 2018 standard
    """
    labels[labels == 6.0] = 5.0  # very low clouds
    labels[labels == 8.0] = 6.0  # low clouds
    labels[labels == 10.0] = 7.0  # middle clouds
    labels[labels == 12.0] = 8.0  # high opaque clouds
    labels[labels == 14.0] = 9.0  # very high opaque clouds
    labels[labels == 19.0] = 10.0  # fractional clouds
    labels[labels == 15.0] = 11.0  # high semitransparent thin clouds
    labels[labels == 16.0] = 12.0  # high semitransparent moderatly thick clouds
    labels[labels == 17.0] = 13.0  # high semitransparent thick clouds
    labels[labels == 18.0] = 14.0  # high semitransparent above low or medium clouds
    # missing: 15:  High semitransparent above snow/ice
    return labels


def switch_2013(labels):
    """
    maps netcdf cloud types from the 2018 standard to the 2013 standard
    """
    labels[labels == 15.0] = 18.0  # high semitransparent above snow/ice
    labels[labels == 14.0] = 18.0  # high semitransparent above low or medium clouds
    labels[labels == 13.0] = 17.0  # high semitransparent thick clouds
    labels[labels == 12.0] = 16.0  # high semitransparent moderatly thick clouds
    labels[labels == 11.0] = 15.0  # high semitransparent thin clouds
    labels[labels == 10.0] = 19.0  # fractional clouds
    labels[labels == 9.0] = 14.0  # very high opaque clouds
    labels[labels == 8.0] = 12.0  # high opaque clouds
    labels[labels == 7.0] = 10.0  # middle clouds
    labels[labels == 6.0] = 8.0  # low clouds
    labels[labels == 5.0] = 6.0  # very low clouds

    return labels
