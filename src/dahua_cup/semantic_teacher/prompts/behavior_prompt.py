def build_prompt(student_result=None):

    prompt = """
你是一名高校智慧校园行为分析专家。

你的任务是分析校园监控视频中的学生行为。

需要区分以下六类行为：

1. 正常行走
2. 正常奔跑
3. 嬉戏追逐
4. 嬉戏推搡
5. 冲突追逐
6. 冲突推搡

请重点观察以下语义特征：

① 是否存在持续追逐

② 双方轨迹是否折返

③ 是否出现攻击动作

④ 是否存在防御姿态

⑤ 接触力度是否较大

⑥ 是否持续发生身体接触

⑦ 周围人员是否参与或围观

不要只依据速度进行判断。

请综合行为意图进行分析。

最后必须严格输出 JSON。

JSON格式如下：

{
    "label":"",
    "confidence":0.0,
    "reason":""
}

其中：

label只能是：

正常行走
正常奔跑
嬉戏追逐
嬉戏推搡
冲突追逐
冲突推搡

confidence范围：

0~1

reason一句话即可。

"""

    # ===== 如果没有小模型结果 =====
    if student_result is None:
        return prompt

    # ===== 有小模型结果 =====
    prompt += f"""

----------------------------------------
轻量模型(ST-GCN)预测结果：

预测类别：{student_result.get("label","未知")}

置信度：{student_result.get("confidence",0)}

轨迹特征：{student_result.get("trajectory","未知")}

速度：{student_result.get("speed","未知")}

接触情况：{student_result.get("contact","未知")}

请结合以上信息和视频重新分析。

如果你认为轻量模型判断错误，可以纠正。

最终输出JSON。
"""

    return prompt