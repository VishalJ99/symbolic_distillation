def normalise_sim_filename(filename):
    """
    Normalize the filename to ensure consistent comparison.
    Removes the directory and initial part of the filename.
    """
    # Remove directory if present
    if "/" in filename:
        filename = filename.split("/")[-1]
    # Normalize prefix differences
    return filename.replace("test_", "").replace("colab_", "")
