from pathlib import Path
import globox


def main() -> None:
  path = Path("/path/to/annotations/")  # Where the .txt files are
  save_file = Path("coco.json")

  annotations = globox.AnnotationSet.from_yolo("/home/tarun/NumberTheory/TradeFinance/table-finetuning/CI_table_annotate_214",
    image_folder="/home/tarun/NumberTheory/TradeFinance/table-finetuning/CI_table_annotate_214/Images")
  annotations.save_coco(save_file, auto_ids=True)


if __name__ == "__main__":
    main()