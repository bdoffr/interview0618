# 标准库导入
import os
import re
import json
import logging  # 引入日志模块
from typing import Dict, List, Any, Optional

# 第三方库导入
from asgiref.sync import sync_to_async
from openai import OpenAI
from django.http import JsonResponse
from django.views import View
from django.shortcuts import render
import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from langchain.schema import HumanMessage, AIMessage  # 保持使用
from langchain.prompts import PromptTemplate
from datetime import datetime, timedelta

# 本地应用/库特定导入
from .models import Repository, RepositoryDetail

# ================== 配置区 ==================

# # 数据库配置
# SQLITE_DB_PATH = "/root/test/phoenix-develop/phoenix-develop/db.sqlite3"
# TABLE_NAME = "ai_customer_repositorydetail"
logger = logging.getLogger('django')
# 配置向量数据库参数
VECTOR_DB_DIR = "/root/autodl-tmp/phoenix/chroma_db"
# VECTOR_DB_DIR = "/root/rag/chroma/chroma_db"
COLLECTION_NAME = "multimodal_index"
EMBED_MODEL_PATH = "/root/autodl-tmp/m3e-base"

# 查询参数
SIMILARITY_TOP_K = 3
SIMILARITY_THRESHOLD = 0.6
TEMPERATURE = 0.3


def home(request):
    return render(request, "home.html", {})


def repository_index(request):
    repositories = Repository.objects.all()
    context = {
        "repositories": repositories
    }
    return render(request, "customer/project_index.html", context)


def repository_detail(request, pk, name, description):
    repositories = RepositoryDetail.objects.filter(repository_id=pk)
    context = {
        "name": name,
        "description": description,
        "repositories": repositories
    }
    return render(request, "customer/project_detail.html", context)


# 调用大模型生成回复
class FunctionCall:
    def __init__(self):
        """
        初始化 OpenAI 客户端、工具定义并加载意图
        """
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        # 工具定义
        self.tools = [
            {"type": "function",
             "function": {
                 "name": "execute_intent_analysis",
                 "description": "分析用户输入以识别其意图.可用意图包括:transport_status_query (查询发货状态), query (通用查询/查询商品详情), product_detail_query (查询订单中的商品详情), cancel_order (取消订单), change_address (修改地址), invoicing (开发票), check_invoice (查询发票), query_express (查询快递), logistics_track (查询物流轨迹详情), query_gift (查询赠品), transfer_manual (转人工), classic_style (经典款推荐), family_style (家庭装推荐), portable_style (便携装推荐), zuikaku (瑞鹤款推荐), roof_package (屋顶包推荐), preface_to_lanting (兰亭序款推荐), unknown_intention (未知意图)",
                 "parameters": {
                     "type": "object",
                     "properties": {
                         "user_content": {
                             "type": "string",
                             "description": "需要分析意图的用户原始输入"
                         }
                     },
                     "required": ["user_content"]
                 }
             }
             },
            {"type": "function",
             "function": {"name": "judge",
                          "description": "判断产品规格是否相等",
                          "parameters": {"type": "object",
                                         "properties": {
                                             "product_detail": {"type": "string", "description": "产品详情"},
                                             "erp_product_detail": {"type": "string",
                                                                    "description": "erp查询的产品详情"}},
                                         "required": ["product_detail", "erp_product_detail"]}}},
            {"type": "function", "function": {"name": "logistics_track", "description": "判断物流轨迹",
                                              "parameters": {"type": "object", "properties": {
                                                  "express_list": {"type": "list", "description": "物流详情列表"}},
                                                             "required": ["express_list"]}}}
        ]
        # 加载用于意图识别的提示模板
        self._load_intent_prompts()

    def _load_intent_prompts(self):
        """
        从json文件加载意图提示模板
        """
        # 默认回退提示
        self.intent_prompt_template = "可用的意图是: [transport_status_query, query, product_detail_query]"
        self.instruction_template = "User input: {user_content}\nDetermine the intent from the available intents."

        prompt_file_path = '/root/autodl-tmp/phoenix/data/intent_prompt.json'  # Critical: Ensure this path is correct
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            prompt_config = json.load(f)
        # Ensure the keys exist in your JSON file structure
        self.intent_prompt_template = prompt_config.get('prompt_template', {}).get('intents',
                                                                                   self.intent_prompt_template)
        self.instruction_template = prompt_config.get('prompt_template', {}).get('instruction',
                                                                                 self.instruction_template)

    def _perform_actual_intent_classification(self, text_to_classify: str) -> str:
        """
        Performs the actual intent classification using a specialized prompt and an LLM call.
        'execute_intent_analysis'工具分类调用核心逻辑
        """
        # 使用加载模板和分类文本格式化提示
        instruction = self.instruction_template.format(user_content=text_to_classify)
        full_prompt_for_classification = f"{self.intent_prompt_template}\n{instruction}"
        try:
            response = self.client.chat.completions.create(
                model="qwen-plus-latest",  # Or your preferred model for this classification task
                messages=[{"role": "user", "content": full_prompt_for_classification}],
                temperature=0.1,  # 对于分类任务，较低的温度通常更好
                # No 'tools' or 'tool_choice' here for a direct classification response
            )
            classified_intent = response.choices[0].message.content
            return classified_intent.strip() if classified_intent else "意图识别未生效"
        except Exception as e:
            logger.error(f"调用LLM进行意图分类时出错: {e}")
            return "意图分类时出错"

    def get_user_intent(self, user_content: str) -> str:
        """
        Orchestrates the process of recognizing user intent.
        1. Makes an initial LLM call, configured to use the 'execute_intent_analysis' tool.
        2. If the LLM decides to call the tool, this function extracts arguments and then
           invokes '_perform_actual_intent_classification' to get the intent.
        3. Returns the recognized intent string.
        """
        initial_messages_for_llm = [{"role": "user", "content": user_content}]

        try:
            # 步骤1: 调用LLM, 提供'execute_intent_analysis'工具定义
            # 强制使用此'execute_intent_analysis'
            intent_llm_response = self.client.chat.completions.create(
                model="qwen-plus-latest",
                messages=initial_messages_for_llm,
                tools=self.tools,  # Define the tool(s) available
                tool_choice={"type": "function", "function": {"name": "execute_intent_analysis"}}  # Force this tool
            )
            message_from_llm = intent_llm_response.choices[0].message

            # 步骤2: 检查LLM是否对预期工具做出响应
            if message_from_llm.tool_calls:
                for tool_call in message_from_llm.tool_calls:
                    if tool_call.function.name == "execute_intent_analysis":
                        # Extract arguments provided by the LLM for our tool
                        tool_arguments = json.loads(tool_call.function.arguments)
                        text_for_intent_analysis = tool_arguments.get("user_content")
                        final_intent = self._perform_actual_intent_classification(text_for_intent_analysis)
                        return final_intent

            # This part should ideally not be reached if tool_choice is effective.
            # It means the LLM didn't make the expected tool call.
            warning_message = "LLM没有进行预期的“execute_intent_analysis”工具调用。"
            if message_from_llm.content:
                warning_message += f" LLM response: {message_from_llm.content}"
            logger.warning(warning_message)
            return "LLM未按预期调用意图识别工具"

        except Exception as e:
            error_message = f"get_user_intent发生意外错误: {e}"
            logger.error(error_message)
            return error_message

    def product_spec(self, query):
        prompt = (f"""请对比本店产品,判断用户的输入中有无相应规格
本店产品:
- 黑金东方苏打(天然苏打水)
    - 经典款 475ml
    - 家庭装 1.5L
    - 便携装 330ml
    - 踏青装 475ml
    - 瑞鹤限定款 475ml
- 东方白桦(白桦树汁)
    - 屋顶包 330ml
用户输入：
"{query}"
有则返回TRUE,无则返回FALSE,不要有其他的话
""")
        response = self.client.chat.completions.create(
            model="qwen-plus-latest",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ],
            # tools=self.tools,
            # tool_choice={"type": "function", "function": {"name": "product_spec"}}
        )
        return response.choices[0].message.content

    # product_detail: 不是我之前买的都是475ml啊，这回咋变300多了
    # erp_product_detail: 飞飞给的
    def judge(self, product_detail, erp_product_detail):
        prompt = (f"""判断以下两者规格是否相等
用户想要的产品规格:{product_detail}
查询到的产品规格:{erp_product_detail}
相等则返回TRUE,不等则返回FALSE,不要有其他的话
""")
        response = self.client.chat.completions.create(
            model="qwen-plus-latest",
            messages=[
                {"role": "system", "content": prompt}
            ],
        )
        return response.choices[0].message.content

    def logistics_track(self, express_list):
        prompt = (f"""判断物流轨迹是否异常
- 物流轨迹:{express_list}
- 检测地理位置
    - 当物流轨迹中出现来回跳跃时为异常(如A→B→A)
- 只返回结果
""")
        response = self.client.chat.completions.create(
            model="qwen-plus-latest",
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content


# ================== 查询优化 ==================
@sync_to_async
def optimize_response(intent,
                      input_text: str,
                      history=None,
                      retrieved_docs: list = None) -> dict:
    logger.info(f"intent: {intent}")
    data = {
        "user_content": "",
        "search": {"query_order": False, "query_express": False, "express_detail": False, "query_gift": False,
                   "cancel_order": False,
                   "change_address": False},
        "invoice": "",
        "recommend": "",
        "transfer_manual": False
    }
    # if history is None:
    #     history = []
    if bool(re.search(r'(?<!\d)\d{19}(?!\d)', input_text)):
        if history:
            content_list = [msg["content"].strip() for msg in history if msg["role"] == "user"][::-1]
            # ['我要买水', '最后一个', '查发票']
            for last_history in content_list:
                intent_handler = FunctionCall()
                last_intent = intent_handler.get_user_intent(user_content=last_history)
                if last_intent:
                    # query = json.loads(tool_calls[0]["function"]["arguments"])["query"]
                    # response = FunctionCall().intent_recognition(query)
                    if last_intent == "invoicing":
                        intent += "invoicing"
                        data["invoice"] = "issue"
                        data["user_content"] = "收到您的订单编号,这面为您推送开票链接❤️"
                        break
                    if last_intent == "check_invoice":
                        intent += "check_invoice"
                        data["invoice"] = "query"
                        data["user_content"] = "收到您的订单编号,这面为您查询一下发票❤️"
                        break
                    if last_intent == "query_express":
                        data["search"]["query_express"] = True
                        data["user_content"] = "收到您的订单编号,这面为您查询一下快递❤️"
                        break
                    if last_intent == "logistics_track":
                        data["search"]["express_detail"] = True
                        data["user_content"] = "收到您的订单编号,这面为您查询一下物流信息❤️"
                        break
                    if last_intent in {"transport_status_query", "product_detail_query", "query"}:
                        data["search"]["query_order"] = True
                        data["user_content"] = "收到您的订单编号,这面为您查询一下商品详情❤️"
                        break
                    if last_intent == "cancel_order":
                        data["search"]["cancel_order"] = True
                        data["user_content"] = "宝儿,您点击下处理退款就可以啦❤️"
                        break
                    if last_intent == "change_address":
                        data["search"]["change_address"] = True
                        data["user_content"] = "收到您的订单编号,这面为您修改地址❤️"
                        break
                    if last_intent == "query_gift":
                        data["search"]["query_gift"] = True
                        data["user_content"] = "收到您的订单编号,这面为您查询赠品❤️"
                        break
                    if last_intent == "unknown_intention":
                        continue
            return data
    docs_info = "以下是知识库中的相关资料(字段:值)：\n"
    if retrieved_docs:
        for doc in retrieved_docs:
            # 过滤非字典元素
            if not isinstance(doc, dict):
                # 如果是字符串,尝试解析为字典
                if isinstance(doc, str):
                    try:
                        doc = json.loads(doc)
                    except json.JSONDecodeError:
                        continue  # 解析失败则跳过
                else:
                    continue  # 其他类型直接跳过
            modified_doc = {}
            for key, value in doc.items():
                # # 替换字段名
                # if key == "index":
                #     modified_key = "用户"
                # elif key == "content":
                #     modified_key = "客服"
                # elif key == "product":
                #     modified_key = "产品"
                # elif key == "price":
                #     modified_key = "价格"
                # else:
                modified_key = key  # 保留其他字段名不变
                modified_doc[modified_key] = value
            # 将修改后的字段写入 docs_info
            for key, value in modified_doc.items():
                docs_info += f"- {key}: {value}\n"
    # data = {
    #     "user_content": "",
    #     "search": {"query_order": False, "query_express": False, "express_detail": False, "query_gift": False,
    #                "cancel_order": False,
    #                "change_address": False},
    #     "invoice": "",
    #     "recommend": "",
    #     "transfer_manual": False
    # }
    logger.info(f"知识库检索: {docs_info}")
    template = f"""你是抖音电商平台的客服蛋蛋
品牌与产品:
- 品牌名称:长生之脉
- 售卖渠道：抖音，京东，淘宝，拼多多，线下高端商超
    - 抖音平台订单的消息通知都在"我的订单"里面
- 产品分类:
    - 黑金东方苏打(天然苏打水)
        - 经典款 475ml 12瓶*4箱共189元/12瓶*6箱共279元
        - 家庭款 1.5L 6瓶*4箱共258元
        - 便携款 330ml 12瓶1箱共49元/12瓶*3箱共119元/12瓶*6箱共219元
        - 踏青款 475ml
        - 瑞鹤限定款 475ml 12瓶*4箱共269元
    - 东方白桦(白桦树汁)
        - 屋顶包 330ml 8瓶*1箱共69元/16瓶*2箱共129元/24瓶*3箱共189元
    - 长生之脉最新款
        - 兰亭序联名限定款
- 试饮装:即2瓶装、试喝装、尝鲜装
- 机制:商品规格和当前促销活动
- 海哥兰亭序促销活动
    - 拍1单:179元*4箱 赠送一份长生之脉龙井春芽茶包
    - 拍2单:358元*8箱 赠送一份长生之脉龙井春芽茶包和长生之脉兰亭联名折扇1把
    - 拍4单:716元*16箱 顾客拍4单后先到手4箱水+兰亭新品水卡一张+长生之脉龙井茶包1盒,后续顾客可以通过水卡兑换兰亭限定的12箱水+长生之脉兰亭联名折扇1把+长生之脉兰亭联名鎏银直筒杯1只
        - 赠送一个长生之脉龙井春芽茶包,长生之脉兰亭联名折扇1把,长生之脉兰亭联名鎏银直筒杯1只,一张水卡
    - 拍6单:1074元*24箱 顾客拍6单后先到手4箱水+兰亭新品水卡一张+长生之脉龙井茶包1盒,后续顾客可以通过水卡兑换兰亭限定的20箱水+长生之脉兰亭联名折扇1把+长生之脉兰亭联名鎏银直筒杯1只+长生之脉兰亭联名鎏银三才盖碗1只
        - 赠送一个长生之脉龙井春芽茶包,长生之脉兰亭联名折扇1把+长生之脉兰亭联名鎏银直筒杯1只+长生之脉兰亭联名鎏银三才盖碗1只,一张水卡
- 官方直播间/店铺兰亭序促销活动
    - 拍1单:109元*2箱 赠送一份长生之脉龙井春芽茶包
    - 拍1单:189元*4箱 赠送一份长生之脉龙井春芽茶包
    - 拍1单:279元*6箱 赠送一个钛茶碗
    - 拍1单:828元*18箱 赠送一个银杯,一把长生之脉兰亭联名折扇,一份长生之脉龙井春芽茶包
    - 拍1单:1099.2元*24箱 赠送一个银杯,一把长生之脉兰亭联名折扇,一份长生之脉龙井春芽茶包,一个长生之脉兰亭联名鎏银三才盖碗
    - 拍2单:558元*12箱 赠送一个钛茶碗,一个白桦树汁2瓶试喝装
- 交互规范:
    - 回复风格:亲切自然,温柔而不失专业度,说话体贴,用口语化的说话方式进行回复(最多不超过20个字)
    - 重复场景:遇到历史对话时可变换句式
    - 称谓使用:统一称"宝/宝儿/宝宝"(根据语境灵活选择)
    - 不可以更改商品规格
    - 无论是多少年的老顾客,也没有赠送的活动
    - 回复逻辑符合中国人沟通逻辑
    - 禁止反问用户的需求,仅回复{input_text},不引导客户继续提问,不可以帮用户下单,禁止使用"请随时问我""可以随时提问""请随时联系我""有其他需要帮忙的吗？"
    - 禁止回复"哎呀", "嘿嘿", "哈哈", "嘻嘻"等词汇
    - 禁止以"您好","你好","您好呀","哦"开头
    - 禁止夸奖用户
    - 商品划算问题优先计算单瓶价格，按照单瓶价格进行判断并回复给用户
    - 计算价格如果不能整除则保留小数点儿后两位，并且不进行四舍五入
    - 不得承诺赠品、福利或私信回复,所有回复需符合抖音平台政策
    - 禁止泄露用户隐私,发现敏感信息立即提醒删除
    - 回复中提到"微信"时用"绿泡泡"代替
- 物流:
    - 快递默认京东或顺丰,送货上楼,不支持指定快递
    - 到货时间:全国多仓发货,一般地区2-5天,偏远地区会稍长一些,非预售商品下单后24-48小时内发货
- 核心能力:
    - 产品反馈处理流程:
        - 产品日期问题:当用户询问产品生产日期时,我们应回复给用户"因为是多仓发货的，无法确保准确更长日期的，但是目前所有商品都是2025年近几个月的新鲜好日期哦"
        - 口感类问题:天然苏打水反馈口感问题是体质原因,体质不同口感会有偏差,白桦树汁如果反馈酸了,是天气原因导致变质,应该及时给用户进行补偿赔付
        - 产品质量问题:瓶身有划痕等,是工厂生产的问题,明确顾客的诉求,并进行安抚
    - 产品价格反馈:当用户反馈购买价格低于日常售价时,告知用户是因为平台发放的随机优惠券 不定时会发放给优质的客户
    - 多平台比价:抖音的价格是最合适的，低于京东，淘宝，拼多多。
    - 水王李大海相关问题:水王李大海（海哥）是抖音水饮头部,好的产品需要优秀的人来推广,我们找海哥推广正是因为他是抖音水饮头部,能够更好的传播品牌。海哥价格便宜是因为找海哥推广,限时的福利活动,可以放心购买的。
    - 顾客询问产品配料表或者矿物质成分时，如果顾客没有明确说是哪个产品，优先询问顾客想要咨询的是哪款产品，再进行后续解答
- 对话管理:
    - 收到问候语时(如 "您好/你好/哈喽"),回复"您好呀~我是客服蛋蛋,有什么可以帮助您的吗？"
    - 遇到无关话题时回复"蛋蛋现在只能聊长生之脉相关话题哦~"
- 请根据上述规则及下列信息回答用户问题:
    - {docs_info}
"""
    # prompt_path = '/root/autodl-tmp/phoenix/data/intent_prompt.json'  # Critical: Ensure this path is correct
    # with open(prompt_path, 'r', encoding='utf-8') as f:
    #     template_data = json.load(f)
    # template_str = template_data["prompt_template"]
    # 创建 PromptTemplate 实例
    # logger.info(f"template: {template}")
    prompt = PromptTemplate(
        input_variables=["input"],
        template=template
    )
    client = OpenAI(
        # 若没有配置环境变量,请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus-latest",
        messages=[{"role": "system",
                   "content": prompt.format(input=input_text)}] + history + [
                     {"role": "user", "content": input_text}],
        temperature=TEMPERATURE
    )
    # 获取提示token数
    prompt_tokens = completion.usage.prompt_tokens
    # 获取补全token数
    completion_tokens = completion.usage.completion_tokens
    # 获取总token数
    total_tokens = completion.usage.total_tokens
    logger.info(f"提示Token数: {prompt_tokens}, 补全Token数: {completion_tokens}, 总Token数: {total_tokens}")
    content = completion.choices[0].message.content
    data["user_content"] = content

    if intent == "transport_status_query":
        data["user_content"] = "宝儿稍等~辛苦提供订单编号,现在为您查询发货状态呢❤️"
    elif intent == "product_detail_query":
        data["user_content"] = "宝儿稍等~辛苦提供订单编号,为您核对商品详情呢❤️"
    elif intent == "query":
        data["user_content"] = "宝儿稍等~辛苦提供订单编号,为您查询下商品详情呢❤️"
    elif intent == "cancel_order":
        data["user_content"] = "宝儿辛苦提供订单编号,为您取消订单呢❤️"
    elif intent == "change_address":
        data["user_content"] = "宝儿辛苦提供订单编号,为您推送修改链接呢❤️"
    elif intent == "invoicing":
        data["user_content"] = "宝儿辛苦提供订单编号,麻烦您订单先确认收货，小客服为您推送开票链接哦❤️"
    elif intent == "check_invoice":
        data["user_content"] = "宝儿辛苦提供订单编号,这面为您查询一下发票呢❤️"
    elif intent == "query_express":
        data["user_content"] = "宝儿辛苦提供订单编号,这面为您查询一下快递呢❤️"
    elif intent == "logistics_track":
        data["user_content"] = "宝儿辛苦提供订单编号,这面为您查询一下物流信息呢❤️"
    elif intent == "query_gift":
        data["user_content"] = "宝儿辛苦提供订单编号,这面为您查询一下赠品呢❤️"
    elif intent == "transfer_manual":
        data["user_content"] = "这边看到了您的情况,为了更高效快速的处理您的问题,这边帮您单独转接专业的售后专员为您进行处理宝儿❤️"
        data["transfer_manual"] = True
    elif intent == "classic_style":
        data["recommend"] = "拍一发四"
    elif intent == "family_style":
        data["recommend"] = "1.5L*6瓶*4箱/共24瓶"
    elif intent == "portable_style":
        data["recommend"] = "330ml*72瓶/共六箱"
    elif intent == "zuikaku":
        data["recommend"] = "瑞鹤"
    elif intent == "roof_package":
        data["recommend"] = "330ML*8瓶*3箱"
    elif intent == "preface_to_lanting":
        data["recommend"] = "兰亭序"
    logger.info(f"data:{data}")
    return data


# - 明确用户诉求后回复"稍后由专属售后专员处理~"
@sync_to_async
def image_response(history, input_image: str, intent: str) -> dict:
    template = f"""你是抖音电商平台的客服蛋蛋
品牌与产品:
- 品牌名称:长生之脉
- 交互规范:
    - 回复风格:亲切自然,温柔而不失专业度,说话体贴,用口语化的说话方式进行回复(最多不超过20个字)
    - 重复场景:遇到历史对话时可变换句式
    - 称谓使用:统一称"宝/宝儿/宝宝"(根据语境灵活选择)
    - 不可以更改商品规格
    - 无论是多少年的老顾客,也没有赠送的活动
    - 回复逻辑符合中国人沟通逻辑
    - 禁止反问用户的需求,仅回复当前对话,不引导客户继续提问,不可以帮用户下单,禁止使用"请随时问我""可以随时提问""请随时联系我""有其他需要帮忙的吗？"
    - 禁止回复"哎呀", "嘿嘿", "哈哈", "嘻嘻"等词汇
    - 禁止以"您好","你好","您好呀","哦"开头
    - 禁止夸奖用户
    - 不得承诺赠品、福利或私信回复,所有回复需符合抖音平台政策
    - 禁止泄露用户隐私,发现敏感信息立即提醒删除
    - 回复中提到“微信”时用绿泡泡代替

- 当图片中存在瓶子时,请识别图片中瓶身主体,不包括背景中的其他物品或人:
    - 兰亭序联名限定款
        书法字体:瓶身标签上印有"兰亭序"书法字体的饮品
        Logo:识别带有长生之脉和兰亭书法博物馆联名Logo的饮品
        系列名称:标签上写有"长生之脉""东方苏大"字样,其下方配有对应的英文"THE ORCHID PAVILION"的饮品
        瓶身印章:标签底部写有篆书"长生之脉"字样的饮品
        容量:一瓶净含量475ml的兰亭序饮品
    - 黑金东方苏打(天然苏打水)
        - 瓶身配色:瓶身配色为黑色和金色
        - 容量:瓶装水的容量为475ml、330ml或1.5L
        - Logo:瓶身有“东方苏打”或“长生之脉”的LOGO字样
    - 识别是否为自家产品,判断产品类别
- 当图片中存在订单编号时识别订单编号
根据上述规则,分析提供的图片并生成回答：
"""
    # - 售后问题：识别图片里顾客购买的产品是否出现包装破损,漏水,结冰等问题,优先安抚用户,然后转接专业的售后人员进行处理
    # 创建 PromptTemplate 实例
    prompt = PromptTemplate(
        input_variables=["intent"],
        template=template
    )
    client = OpenAI(
        # 若没有配置环境变量,请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-vl-max-2025-04-08",
        messages=[
                     {"role": "system",
                      "content": prompt.format(intent=intent)}] + history +
                 [{"role": "user",
                   "content": [
                       # {"type": "text", "text": "有瓶水坏了"},
                       {"type": "image_url", "image_url": input_image}
                   ]}
                  ],
        temperature=TEMPERATURE
    )
    data = {}
    try:
        data["assistant"] = completion.choices[0].message.content
        # return data
    except Exception as e:
        logger.error(f"API请求失败: {str(e)}")
        data["assistant"] = "系统繁忙,请稍后再试"
    return data


memories: Dict[str, List] = {}


# 获取对话历史
@sync_to_async
def get_chat_history(nick_name: str) -> List[Dict[str, str]]:
    """
    获取格式化后的对话历史。
    返回格式: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    if nick_name not in memories:
        return []
    chat_history = memories[nick_name][-10:]
    # return memories[nick_name][-10:]
    # 格式化历史记录为可读文本
    formatted_history = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            formatted_history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            formatted_history.append({"role": "assistant", "content": msg.content})
    return formatted_history

@sync_to_async
def update_memory(nick_name, user_content, optimized_result):
    """
    更新对话历史。
    """
    if nick_name not in memories:
        memories[nick_name] = []

    # 添加用户消息和 AI 回复
    memories[nick_name].append(HumanMessage(content=user_content))
    memories[nick_name].append(AIMessage(content=optimized_result))


class RPAIndexView(View):
    # def __init__(self):
    #     super().__init__()
    #     self.tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)  # 可选参数,允许不传入

    async def post(self, request):
        if request.method == "POST":
            # 解析请求参数
            body = json.loads(request.body)
            user_content = body.get("user_content")
            nick_name = body.get("nickname")
            problem_type = body.get("type")
            product_name = body.get("product_name")
            logger.info(f"request: {body}")
            # # 根据当前 user_content 更新 MAX_HISTORY_TOKENS
            # self.MAX_HISTORY_TOKENS = 8000 - len(self.tokenizer.encode(user_content)) - 50
            if not nick_name:
                return JsonResponse({"error": "无用户昵称"}, status=400)
            # 初始化 memory
            if nick_name not in memories:
                memories[nick_name] = []
            # 获取历史记录
            history = await get_chat_history(nick_name)
            # # 截断历史记录以适应最大token限制
            # truncated_history = self.truncate_history(history)
            # user_pattern = re.compile(r'^用户:\s*(.*)', re.MULTILINE)
            # 意图识别
            intent_handler = FunctionCall()
            intent = intent_handler.get_user_intent(user_content=user_content)
            logger.info(f"用户意图: {intent}")
            # 初始化Chroma客户端
            chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
            chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_PATH)
            # documents = SimpleDirectoryReader("/root/phoenix/data").load_data()
            Settings.llm = None
            # 构建索引和查询引擎
            index = VectorStoreIndex.from_documents(
                documents=[],
                storage_context=storage_context,
                embed_model=embed_model
            )
            # 相似度阈值后处理器
            # processor = SimilarityPostprocessor(similarity_cutoff=SIMILARITY_THRESHOLD)
            # query_engine = index.as_query_engine(similarity_top_k=SIMILARITY_TOP_K, node_postprocessors=[processor])
            query_engine = index.as_query_engine(similarity_top_k=SIMILARITY_TOP_K,
                                                 similarity_cutoff=SIMILARITY_THRESHOLD)
            response = query_engine.query(user_content)
            retrieved_docs = []
            for node in response.source_nodes:
                text = node.node.text
                try:
                    data = json.loads(text.replace('\ufeff', ''))
                    retrieved_docs.append(data)
                except:
                    retrieved_docs.append(text)
            if problem_type == "text":
                optimized_result = await optimize_response(
                    history=history,  # 历史消息
                    intent=intent,
                    input_text=user_content,  # 用户输入
                    retrieved_docs=retrieved_docs  # rag回答
                )
                await update_memory(nick_name, user_content, optimized_result["user_content"])
                logger.info(f"response: {optimized_result}")
                return JsonResponse(optimized_result, status=200)
            elif problem_type == "image":
                optimized_result = await image_response(
                    history=history,
                    intent=intent,
                    input_image=user_content,
                    # retrieved_docs=[]
                )
                # data["user_content"] = optimized_result
                await update_memory(nick_name, user_content, optimized_result["assistant"])
                logger.info(f"response: {optimized_result}")
                return JsonResponse(optimized_result, status=200)
            elif problem_type == "query":
                response_dict = {
                    "user_content": "",
                    "search": {
                        "query_order": False,
                        "query_express": False,
                        "cancel_order": False,
                        "change_address": False
                    },
                    "invoice": "",
                    "recommend": "",
                    "transfer_manual": False
                }
                content_list = [msg["content"].strip() for msg in history if msg["role"] == "user"][::-1]
                if product_name == "":
                    response_dict["user_content"] = "宝儿, 查询不到您的订单, 您再核对下订单编号呢？"
                    await update_memory(nick_name, user_content, response_dict["user_content"])
                    logger.info(f"response: {response_dict}")
                    return JsonResponse(data=response_dict, status=200)
                else:
                    for last_history in content_list:
                        # intent_handler = FunctionCall()
                        # last_intent = intent_handler.get_user_intent(user_content=last_history)
                        intent = intent_handler.get_user_intent(user_content=last_history)
                        logger.info(f"intent: {intent}, last_history: {last_history},")
                        if intent == "product_detail_query":
                            product_spec = FunctionCall().product_spec(last_history)
                            if product_spec == "TRUE":
                                # 判断拍错/发错
                                final_response = FunctionCall().judge(last_history, product_name)
                                if final_response == "TRUE":
                                    response_dict[
                                        "user_content"] = f"宝儿,这边看到您的订单是{product_name},辛苦您再核对下当时拍的订单"
                                else:
                                    response_dict["transfer_manual"] = True
                                    response_dict["user_content"] = "为了更好更高效的解决您的问题,这边帮您转接专业的售后专业去帮您进行处理哈"
                            else:
                                response_dict[
                                    "user_content"] = f"宝儿,您想要买的是哪一款规格呀？是{product_name}这款嘛"
                                response_dict["search"]["query_order"] = True
                            # response_dict["product_spec"] = product_spec
                        elif intent == "query":
                            response_dict["user_content"] = f"宝儿,这边看到您的订单是{product_name}"
                        else:
                            continue
                    await update_memory(nick_name, user_content, response_dict["user_content"])
                    logger.info(f"response: {response_dict}")
                    return JsonResponse(data=response_dict, status=200)
            else:
                logger.error("错误的查询类型")
                return JsonResponse(data={"error": "错误的查询类型"}, status=400)


class InvoiceView(View):
    async def post(self, request):
        if request.method == "POST":
            body = json.loads(request.body)
            problem_type = body.get("type")
            nick_name = body.get("nickname")
            invoice_status = body.get("invoice_status")
            is_exist = body.get("is_exist")
            is_expire = body.get("is_expire")
            logger.info(f"request: {body}")
            if not nick_name:
                logger.error(f"无用户昵称")
                return JsonResponse({"error": "无用户昵称"}, status=400)
            response = {"user_content": ""}
            if problem_type == "check_invoice":
                if not is_exist:
                    # - 当"发票详情:"为空时,回复"宝儿,发票正在开呢~"
                    response["user_content"] = "宝儿,订单编号错误,您再核对下订单编号呢？"
                elif not is_expire:
                    response["user_content"] = "宝儿，查看到您的发票正在开具中，请您耐心等待，开具好后会为您发送到您的预留的邮箱中的"
                else:
                    if invoice_status == "待开票":
                        response["user_content"] = "宝儿，查看到您的发票正在开具中，请您耐心等待，开具好后会为您发送到您的预留的邮箱中的"
                    elif invoice_status == "已开票":
                        response["user_content"] = "宝儿,查看到您的发票已经开好了,辛苦您去开发票时预留的邮箱里面查看下"
                    else:
                        response["user_content"] = "发票状态错误"
                response = {"user_content": response["user_content"]}
                await update_memory(nick_name, "查询/开具发票", response["user_content"])
                logger.info(f"response: {response}")
                return JsonResponse(response, status=200)
            # elif problem_type == "invoicing":
            #     optimized_result = optimize_response(
            #         intent="invoicing",
            #         input_text=f"发票详情:{invoice_detail},现在是：{invoice_status}",  # 用户输入
            #     )
            #     return JsonResponse(optimized_result, status=200)
            else:
                logger.error("错误的查询类型")
                return JsonResponse({"error": "错误的查询类型"}, status=400)


class QueryExpressView(View):
    async def post(self, request):
        if request.method == "POST":
            body = json.loads(request.body)
            problem_type = body.get("type")
            nick_name = body.get("nickname")
            is_exist = body.get("is_exist")
            is_expire = body.get("is_expire")
            information = body.get("information")
            time = body.get("time")
            logger.info(f"request: {body}")
            if not nick_name:
                logger.error("response: 无用户昵称")
                return JsonResponse({"error": "无用户昵称"}, status=400)
            response = {"user_content": "", "transfer_manual": False}
            if problem_type == "query_express":
                if not is_exist:
                    # - 当"发票详情:"为空时,回复"宝儿,发票正在开呢~"
                    response["user_content"] = "宝儿,查询不到您的快递,您再核对下订单编号呢？"
                elif not is_expire:
                    response["user_content"] = "宝儿,您的订单目前正在发货准备中,暂时查询不到物流信息，为您催促下物流更新"
                else:
                    if not information:
                        logger.error("information为空")
                        return JsonResponse({"error": "information为空"}, status=400)
                    if not time:
                        logger.error("time为空")
                        return JsonResponse({"error": "time为空"}, status=400)
                    express_time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S") + timedelta(days=2)
                    current_time = datetime.now()
                    if current_time <= express_time:
                        response["user_content"] = f"宝儿,这面查询到您的快递信息:\n{information}"
                    else:
                        response["transfer_manual"] = True
                        response[
                            "user_content"] = f"宝儿,这边看到了您的情况,为了更高效快速的处理您的问题,这边帮您单独转接专业的售后专员为您进行处理"
                await update_memory(nick_name, "查快递", response["user_content"])
                logger.info(f"response: {response}")
                return JsonResponse(response, status=200)
            else:
                logger.error("错误的查询类型")
                return JsonResponse({"error": "错误的查询类型"}, status=400)


class CancelOrderView(View):
    async def post(self, request):
        if request.method == "POST":
            body = json.loads(request.body)
            problem_type = body.get("type")
            nick_name = body.get("nickname")
            is_exist = body.get("is_exist")
            is_expire = body.get("is_expire")
            logger.info(f"request: {body}")
            if not nick_name:
                logger.error(f"无用户昵称")
                return JsonResponse({"error": "无用户昵称"}, status=400)
            response = {"user_content": "", "transfer_manual": False}
            if problem_type == "cancel_order":
                response["user_content"] = "宝儿，这边已经帮您填好预售后申请啦，您点击下处理提交就可以的哦"
                # intent = "cancel_order"
                if not is_exist:
                    response["user_content"] = "宝儿,查询不到您的订单,您再核对下订单编号呢？"
                elif not is_expire:
                    response["user_content"] = "宝儿,这面查看到您的订单正在发货中,现在为您转接售后专员呢"
                    response["transfer_manual"] = True
                await update_memory(nick_name, "取消订单", response["user_content"])
                logger.info(f"response: {response}")
                return JsonResponse(response, status=200)
            else:
                logger.error(f"错误的查询类型")
                return JsonResponse({"error": "错误的查询类型"}, status=400)


class ChangeAddressView(View):
    async def post(self, request):
        if request.method == "POST":
            body = json.loads(request.body)
            problem_type = body.get("type")
            nick_name = body.get("nickname")
            is_exist = body.get("is_exist")
            is_expire = body.get("is_expire")
            logger.info(f"body: {body}")
            if not nick_name:
                logger.error("无用户昵称")
                return JsonResponse({"error": "无用户昵称"}, status=400)
            response = {"user_content": "", "transfer_manual": False}
            if problem_type == "change_address":
                response["user_content"] = "宝儿,这边给您发送修改链接,您填好就可以啦"
                if not is_exist:
                    response["user_content"] = "宝儿,查询不到您的订单,您再核对下订单编号呢？"
                elif not is_expire:
                    response["transfer_manual"] = True
                    response["user_content"] = "宝儿,查看到您的订单已发货,无法修改收货地址的,现在为您转接售后专员呢"
                await update_memory(nick_name, "修改订单", response["user_content"])
                logger.info(f"response: {response}")
                return JsonResponse(response, status=200)
            else:
                logger.error(f"错误的查询类型")
                return JsonResponse({"error": "错误的查询类型"}, status=400)


class QueryGiftView(View):
    async def post(self, request):
        if request.method == "POST":
            body = json.loads(request.body)
            problem_type = body.get("type")
            product_name = body.get("product_detail")
            nick_name = body.get("nickname")
            is_exist = body.get("is_exist")
            logger.info(f"request: {body}")
            if not nick_name:
                logger.error(f"无用户昵称")
                return JsonResponse({"error": "无用户昵称"}, status=400)
            response = {"user_content": "", "transfer_manual": False}
            if problem_type == "query_gift":
                response["user_content"] = ""
                if not is_exist:
                    response["user_content"] = "宝儿,查询不到您的订单,您再核对下订单编号呢？"
                else:
                    if "瑞鹤" in product_name:
                        response["user_content"] = "宝儿,您的赠品茶粉在瑞鹤箱子里哦,那个个头儿比较大,并且有绿色花纹的,您拉开包装就能看到哦"
                    else:
                        response["user_content"] = "这边看到了您的情况,为了更高效快速的处理您的问题,这边帮您单独转接专业的售后专员为您进行处理宝儿❤️"
                        response["transfer_manual"] = True
                await update_memory(nick_name, "赠品查询", response["user_content"])
                logger.info(f"response: {response}")
                return JsonResponse(response, status=200)
            else:
                logger.error(f"错误的查询类型")
                return JsonResponse({"error": "错误的查询类型"}, status=400)


class ExpressDetailView(View):
    async def post(self, request):
        if request.method == "POST":
            body = json.loads(request.body)
            problem_type = body.get("type")
            nick_name = body.get("nickname")
            is_exist = body.get("is_exist")
            is_expire = body.get("is_expire")
            logger.info(f"request: {body}")
            if not nick_name:
                logger.error(f"无用户昵称")
                return JsonResponse({"error": "无用户昵称"}, status=400)
            response = {"assistant": "", "transfer_manual": False}
            if problem_type == "express_detail":
                if not is_exist:
                    # - 当"发票详情:"为空时,回复"宝儿,发票正在开呢~"
                    response["assistant"] = "宝儿,查询不到您的快递,您再核对下订单编号呢?"
                elif not is_expire:
                    response["assistant"] = "宝儿,您的订单目前正在发货准备中,暂时查询不到物流信息,这面为您催促下物流更新"
                else:
                    express_list = body.get("express_list")
                    if not express_list:
                        return JsonResponse({"error": "express_list为空"}, status=400)
                    else:
                        # TODO 对比information物流轨迹,回复用户
                        express_detail = FunctionCall().express_detail(express_list)
                        response["assistant"] = express_detail
                await update_memory(nick_name, "物流轨迹问题", response["assistant"])
                logger.info(f"response: {response}")
                return JsonResponse(response, status=200)
            else:
                logger.error(f"错误的查询类型")
                return JsonResponse({"error": "错误的查询类型"}, status=400)