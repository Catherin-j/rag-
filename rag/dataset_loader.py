def load_dataset(path: str) -> list[dict]:
    """Load dataset lines in the format: Brand - Chemistry."""
    dataset: list[dict] = []

    with open(path, "r", encoding="utf-8") as file_obj:
        lines = file_obj.readlines()

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if "-" not in line:
            # Skip malformed lines instead of crashing startup.
            continue

        brand_raw, chemistry_raw = line.split("-", 1)
        brand = brand_raw.strip().lower()
        chemistry = chemistry_raw.strip()

        if not brand or not chemistry:
            continue

        dataset.append(
            {
                "brand": brand,
                "device": "general",
                "chemistry": chemistry,
                "text": f"{brand} devices use {chemistry} battery chemistry.",
            }
        )

    return dataset