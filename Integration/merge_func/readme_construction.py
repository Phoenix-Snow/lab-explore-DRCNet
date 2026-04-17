from Integration.merge_func.dataset_readme import ReadMeBuilder


def create_dataset_readme(dataset_name, station_3d_details):
    builder = ReadMeBuilder()

    # 1. 封面与数据集名称
    builder.root.text = dataset_name  # 修改根节点名称为数据集名
    builder.add_text(
        "本数据集为面向时空序列分析任务设计的标准化数据集，包含清洗后的原始数据、站点级时序数据、归一化参数及数据划分配置，支持训练、验证及推理阶段的统一数据处理流程。"
        "数据集通过预设的行索引划分（而非训练时划分）确保训练集与验证集的特征覆盖完整性，避免因数据切分导致的模型特征学习不全问题。")

    # 2. 目录结构
    builder.add_header("目录结构", level=2)
    tree_structure = f"{dataset_name}/ \n ├── temporal_resource/                  # 站点级时序数据存储目录 \n"
    for sid in station_3d_details.keys():
        tree_structure = tree_structure \
                         + f" │   ├── {sid}.csv                 # {sid}站点原始时序数据（CSV格式）\n" \
                         + f" │   ├── {sid}.npy                 # {sid}站点的NumPy数组格式数据（用于高效加载）\n" \
                         + f" │   └── {sid}.npz                 # {sid}站点压缩的NumPy数据（含多变量或元数据）\n"
    tree_structure = tree_structure \
                     + " ├── config.json                         # 数据集核心配置文件（含路径、站点信息、序列长度等）\n" \
                     + " ├── domain_split.json                   # 数据集划分配置（按行索引划分训练/验证集）\n" \
                     + " ├── temporal_normalization_params.json  # 时序数据归一化参数（训练阶段计算，用于测试/推理）\n" \
                     + " ├── temporalSpacial_series_cleaned.csv  # 清洗后的完整时空序列数据（所有站点合并，用于对比实验）\n" \
                     + " └── temporalSpacial_series_normal.csv   # 按站点单独归一化的完整时空序列数据（支持跨区域泛化需求）"
    builder.add_code(tree_structure, "text")

    # 3. 核心文件说明
    builder.add_header("核心文件说明", level=2)

    # temporal_resource
    builder.add_header("temporal_resource/", level=3)
    builder.add_text("存储每个站点的独立时序数据，包含三种格式：")
    builder.add_list([
        "**.csv**：原始文本格式，便于人工查看与简单处理。",
        "**.npy**：NumPy二进制格式，支持快速加载与数值计算。",
        "**.npz**：压缩的NumPy格式，可能包含多变量数据或附加元数据。"
    ])
    builder.add_text(f"站点（与站点ID）与地理位置对应关系见config.json中的station_3d字段。")

    # config.json
    builder.add_header("config.json", level=3)
    builder.add_text("关键字段：")
    config_fields = """- **temporal_dataset_save_base**：temporal_resource/目录的基准路径。
- **domain_type**：划分维度（如"province"表示按省级区域划分）。
- **domain**：目标区域列表（如["上海市"]）。
- **station_list**：所有站点的ID列表。
- **station_3d**：站点ID到（经度, 纬度, 高程, 所属区域）的映射（示例："58321199999": [31.143378, 121.805214, 3.96, "上海市"]）。
- **time_series_length**：单条时序的固定长度（如365表示一年365天的数据）。
- **target_domain_ids**：目标预测区域的站点ID列表（如[58367099999]）。"""
    builder.add_code(config_fields, "markdown")
    builder.add_text("【注意】 **temporal_dataset_save_base**等地址配置为本项目路径下，根目录是'dataset'，如果不同需要replace代码更换；"
                     "此外，请注意你生成时使用的操作系统和训练时使用的操作系统。")

    # domain_split.json
    builder.add_header("domain_split.json", level=3)
    builder.add_text(
        "**设计目的**：通过行索引切片（而非随机划分）确保训练集与验证集的时间连续性，避免模型因“前80%训练、后20%验证”的传统划分方式遗漏近年新特征。")
    builder.add_text("**结构**：")
    split_structure = """- **train_list**：各站点训练集的行索引列表。
- **val_list**：各站点验证集的行索引列表"""
    builder.add_code(split_structure, "markdown")

    # temporal_normalization_params.json
    builder.add_header("temporal_normalization_params.json", level=3)
    builder.add_text(
        "**作用**：存储训练阶段计算的归一化参数（最大值、最小值、均值、标准差），确保测试/推理阶段使用相同的归一化逻辑，避免数据分布偏移。")
    builder.add_text("**关键字段**：")
    norm_params = """- **max_values_input/min_values_input**：输入特征的最大值/最小值（用于Min-Max归一化）。
- **mean_values_input/std_values_input**：输入特征的均值/标准差（用于Z-Score归一化）。
- **max_values_output/min_values_output**：输出目标的最大值/最小值。
- **mean_values_output/std_values_output**：输出目标的均值/标准差。"""
    builder.add_code(norm_params, "markdown")

    # temporalSpacial_series_cleaned.csv
    builder.add_header("temporalSpacial_series_cleaned.csv", level=3)
    builder.add_text("所有站点清洗后的合并数据，包含完整时空序列，用于对比实验（如基线模型训练）。")
    builder.add_text("数据格式：每行对应一个时间点，列包含站点ID、时间戳及各特征值。")

    # temporalSpacial_series_normal.csv
    builder.add_header("temporalSpacial_series_normal.csv", level=3)
    builder.add_text(
        "按每个站点单独归一化后的完整时空序列数据，支持跨区域泛化场景（如模型需处理未见过的经纬度区域时，可基于站点级归一化参数调整输入）。")

    # 4. 数据使用说明
    builder.add_header("数据使用说明", level=2)

    builder.add_header("数据加载流程", level=3)
    builder.add_list([
        "读取config.json获取站点ID、序列长度及数据路径。",
        "从temporal_resource/加载目标站点的.npy或.npz文件（推荐NumPy格式以提升效率）。",
        "根据domain_split.json中的train_list/val_list提取对应行索引的时序片段。",
        "使用temporal_normalization_params.json中的参数对输入/输出数据进行归一化（训练阶段需保存参数，测试阶段直接加载）。"
    ])

    builder.add_header("归一化逻辑", level=3)
    builder.add_list([
        "**训练阶段**：基于训练集计算max_values_input、mean_values_input等参数，保存至temporal_normalization_params.json。",
        "**测试/推理阶段**：直接加载预存的归一化参数，对输入数据执行相同的归一化操作（避免使用测试集统计量导致的数据泄露）。"
    ])

    builder.add_header("跨区域泛化支持", level=3)
    builder.add_text(
        "若模型需处理新区域（未见过的经纬度），可使用temporalSpacial_series_normal.csv中按站点归一化的数据，结合station_3d的地理位置信息，通过空间插值或迁移学习策略适配新区域。")

    # 5. 注意事项与版本
    builder.add_header("注意事项", level=2)
    builder.add_list([
        "**数据划分一致性**：务必使用domain_split.json中的行索引进行训练/验证集划分，禁止随机打乱或重新划分，否则可能破坏时间序列的连续性与特征完整性。",
        "**归一化参数复用**：测试/推理阶段必须使用训练阶段保存的temporal_normalization_params.json，禁止重新计算归一化参数。",
        "**站点ID与地理位置**：station_3d字段中的经纬度（如[31.143378, 121.805214]）为WGS84坐标系，高程单位为米，区域名称（如\"上海市\"）需与domain字段一致。"
    ])

    builder.add_header("版本与更新", level=2)
    builder.add_text("**当前版本**：v1.0")
    builder.add_text("**更新日志**：")
    builder.add_list([
        f"**v1.0**：初始版本，包含{len(station_3d_details.keys())}个站点的时序数据，支持按行索引划分与站点级归一化。"
    ])

    builder.add_header("联系方式", level=2)
    builder.add_text("如有疑问或建议，请联系数据集维护者：[{你的邮箱}/{团队名称}]")

    return builder
