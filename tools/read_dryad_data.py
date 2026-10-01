"""Đọc và hiển thị dữ liệu trong bộ dữ liệu Dryad."""

from pathlib import Path
from textwrap import fill

import pandas as pd


# Script nằm trong tools/; dữ liệu và hình xuất ra vẫn ở thư mục gốc project.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (
    REPO_ROOT
    / "Data_DRYAD"
    / "doi_10_5061_dryad_s7h44j150__v20200911"
    / "Dataset-2"
    / "Dataset"
)


def read_and_show_excel(excel_path: Path, rows: int = 10) -> dict[str, pd.DataFrame]:
    """Đọc tất cả các sheet trong file Excel và hiển thị dữ liệu mẫu."""
    if not excel_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {excel_path}")

    workbook = pd.ExcelFile(excel_path)
    print(f"File: {excel_path}")
    print(f"Các sheet: {workbook.sheet_names}\n")

    data = {}
    for sheet_name in workbook.sheet_names:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        data[sheet_name] = df

        print(f"{'=' * 80}\nSheet: {sheet_name}")
        print(f"Kích thước: {df.shape[0]} dòng x {df.shape[1]} cột")
        print("Tên cột:", list(df.columns))
        print("\n10 dòng đầu:")
        print(df.head(rows).to_string(index=False))
        print("\nKiểu dữ liệu:")
        print(df.dtypes.to_string())
        print()

    return data


def select_columns_for_plot(df: pd.DataFrame, limit: int = 12) -> list[str]:
    """Chọn một số cột đại diện để biểu đồ không bị dồn quá nhiều."""
    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    # Với bộ dữ liệu này, các cột mean dễ đọc và trực quan hơn variance/std.
    mean_columns = [
        column for column in numeric_columns if str(column).lower().startswith("mean-")
    ]
    selected = mean_columns[:limit] or numeric_columns[:limit]
    return selected


def create_readable_visualizations(
    df: pd.DataFrame, output_dir: Path, sheet_name: str
) -> None:
    """Tạo biểu đồ dễ đọc thay cho histogram của toàn bộ cột cùng lúc."""
    import matplotlib.pyplot as plt

    columns = select_columns_for_plot(df)
    if not columns:
        print("Không có cột số để vẽ biểu đồ.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_sheet_name = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in sheet_name
    )

    # Mỗi biến có một ô riêng; tên cột dài được xuống dòng để không đè lên nhau.
    n_columns = 3
    n_rows = (len(columns) + n_columns - 1) // n_columns
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(17, 4.2 * n_rows),
        squeeze=False,
    )
    axes = axes.ravel()

    for axis, column in zip(axes, columns):
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        axis.hist(values, bins=25, color="#2E86AB", edgecolor="white", alpha=0.9)
        axis.axvline(
            values.median(),
            color="#D1495B",
            linestyle="--",
            linewidth=1.8,
            label=f"Median: {values.median():.3g}",
        )
        axis.set_title(fill(str(column), width=24), fontsize=10, pad=8)
        axis.set_xlabel("Giá trị", fontsize=9)
        axis.set_ylabel("Số mẫu", fontsize=9)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8, loc="best", frameon=False)
        axis.tick_params(axis="both", labelsize=8)

    # Xóa các ô thừa khi số cột không chia hết cho 3.
    for axis in axes[len(columns) :]:
        axis.remove()

    fig.suptitle(
        f"Phân phối 12 biến đại diện - sheet: {sheet_name}",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    distribution_path = output_dir / f"{safe_sheet_name}_phan_phoi.png"
    fig.savefig(distribution_path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # Boxplot sau khi chuẩn hóa giúp so sánh các biến có thang đo khác nhau.
    standardized = df[columns].apply(pd.to_numeric, errors="coerce")
    standardized = (standardized - standardized.mean()) / standardized.std(ddof=0)
    standardized = standardized.dropna(axis=1, how="all")

    fig, axis = plt.subplots(figsize=(14, max(5, len(standardized.columns) * 0.42)))
    axis.boxplot(
        [standardized[column].dropna() for column in standardized.columns],
        vert=False,
        patch_artist=True,
        boxprops={"facecolor": "#73A2C6", "alpha": 0.8},
        medianprops={"color": "#D1495B", "linewidth": 2},
        whiskerprops={"color": "#555555"},
        capprops={"color": "#555555"},
    )
    axis.set_yticks(range(1, len(standardized.columns) + 1))
    axis.set_yticklabels(
        [fill(str(column), width=28) for column in standardized.columns], fontsize=9
    )
    axis.set_xlabel("Giá trị chuẩn hóa (z-score)")
    axis.set_title("So sánh độ phân tán của các biến đại diện", fontsize=15, pad=14)
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    boxplot_path = output_dir / f"{safe_sheet_name}_boxplot.png"
    fig.savefig(boxplot_path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Đã lưu biểu đồ tại:\n- {distribution_path}\n- {boxplot_path}")


def main() -> None:
    excel_files = sorted(DATA_DIR.glob("*.xlsx"))
    if not excel_files:
        raise FileNotFoundError(f"Không tìm thấy file .xlsx trong: {DATA_DIR}")

    # Dataset hiện có một file Excel chính.
    all_sheets = read_and_show_excel(excel_files[0])

    # Chỉ vẽ sheet đầu tiên để tránh mở quá nhiều cửa sổ biểu đồ.
    first_sheet_name, first_sheet = next(iter(all_sheets.items()))
    create_readable_visualizations(
        first_sheet,
        output_dir=REPO_ROOT / "dryad_visualizations",
        sheet_name=first_sheet_name,
    )


if __name__ == "__main__":
    main()
