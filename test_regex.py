import re

# 测试正则表达式
PREFIXES = ['X', 'Y']
prefixes_upper = {p.upper() for p in PREFIXES}
prefix_pattern = '|'.join(re.escape(p) for p in sorted(prefixes_upper, key=len, reverse=True))
code_re = re.compile(rf'(?<![A-Z])({prefix_pattern})(?=[A-Z0-9-])')

# 测试用例
test_cases = [
    "YKOM0023",
    "配置表NO YMK12",
    "(Y-123)",
    "X16CE-04",
    "JX181170",
    "EX5",
    "品24配置表NO YKOM0023",
    "原12原为X16CE-04.",
]

print("正则表达式测试：")
print(f"Pattern: {code_re.pattern}\n")

for text in test_cases:
    text_up = text.upper()
    m = code_re.search(text_up)
    if m:
        matched = m.group(0)
        prefix = m.group(1)
        print(f"文本: {text_up:30s} -> 匹配: {matched:15s} prefix: {prefix}")
    else:
        print(f"文本: {text_up:30s} -> 无匹配")
