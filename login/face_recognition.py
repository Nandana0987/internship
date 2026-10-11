def recognize_face(frame):
    """
    Replace this with your actual model prediction.
    Must return consistent format.
    """

    # مثال logic (demo only)
    # pretend we detect face

    if frame is None:
        return {"status": "unknown"}

    # TODO: replace with real model
    prediction = "known_user"  # or "unknown"

    if prediction == "known_user":
        return {
            "status": "match",
            "name": "EMP001_Alice"
        }

    return {"status": "unknown"}