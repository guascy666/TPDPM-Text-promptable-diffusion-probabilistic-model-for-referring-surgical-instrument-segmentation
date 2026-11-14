import json

def extract_unique_exp(json_file_path):
    unique_exps = set()
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 遍历所有clip
    for clip_info in data["videos"].values():
        expressions = clip_info["expressions"]
        for item in expressions.values():
            exp_text = item["exp"].strip()
            unique_exps.add(exp_text)
    # 转为列表并排序
    exp_list = sorted(list(unique_exps))
    return exp_list

if __name__ == "__main__":
    # 替换成你的json文件路径
    json_path = r"/media/cygzz/data/guanjl/Code_self/datasets/SGH/combined_output.json"
    result = extract_unique_exp(json_path)
    # 打印所有不同表述
    for idx, exp in enumerate(result, 1):
        print(f"{idx}. {exp}")
    # 也可保存到文本
    with open("unique_exp.txt", "w", encoding="utf-8") as out:
        for exp in result:
            out.write(exp + "\n")