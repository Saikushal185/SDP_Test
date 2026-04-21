from src.dataset_catalog import build_default_dataset_specs, sanitize_dataset_id


def test_default_dataset_specs_include_required_sources():
    specs = build_default_dataset_specs()
    dataset_ids = {spec.dataset_id for spec in specs}

    assert "pd_speech_features_local" in dataset_ids
    assert "kongkon123890_uci_parkinsons_voice" in dataset_ids
    assert "birgermoell_Italian_Parkinsons_Voice_and_Speech" in dataset_ids
    assert "Hahad14_Parkinsons_Disease_Speech" in dataset_ids


def test_kongkon_dataset_has_fallback_candidate():
    specs = build_default_dataset_specs()
    kongkon = next(spec for spec in specs if spec.source_ref == "kongkon123890/uci_parkinsons_voice")

    assert kongkon.fallback_source_ref == "XANJEEV/Parkinson_Classification_Dataset"


def test_sanitize_dataset_id_is_filesystem_safe():
    assert sanitize_dataset_id("birgermoell/Italian_Parkinsons_Voice_and_Speech") == (
        "birgermoell_Italian_Parkinsons_Voice_and_Speech"
    )
