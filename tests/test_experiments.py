from qwen_text2sql.experiments import learning_curve_plan, rank_ablation_plan


def test_learning_curve_drops_sizes_above_total_and_keeps_full() -> None:
    plan = learning_curve_plan(1200)
    assert [cell.train_size for cell in plan] == [250, 500, 1000, None]
    assert plan[-1].name == "n_full"


def test_rank_ablation_is_sorted_and_unique() -> None:
    plan = rank_ablation_plan((16, 4, 16, 8))
    assert [cell.lora_rank for cell in plan] == [4, 8, 16]
