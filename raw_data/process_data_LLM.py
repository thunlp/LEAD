import json
from tqdm import tqdm
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# For better performance, you can choose sample cases from a pool randomly.
prompt = [
    {"role": "system", "content": """你是一个法律专家，能够从案件基本情况中提取关键要素。"""},
    {"role": "user", "content": """我现在有一个数据集，该数据集是用来以案搜案的，但是我的数据集里面的案件基本事实都太长了，我会把这些案件基本情况都发给你，请你帮我简化它们，使其长度大大缩短，例如你可以去掉案件基本事实中并不关键的人名，地点，精简语言等，但不可以删掉对于案件判决有重要作用的事实，如伤情判定等。除了精简之后的案件基本事实之外不要输出任何其他的内容。"""},
    {"role": "assistant", "content": """理解您的需求，请提供更多的案件基本情况，以便我可以帮您进一步精简它们。"""},
    {"role": "user", "content": """2015 年 8 月，巴中鑫金地实业有限公司通过公开竞拍的方式，取得了巴州区回风津桥湖片区国有建设用地使用权，开发建设“英伦庄园”项目。被告人李刚、李均、李洪林、李建、李玉平系该项目用地范围内的巴中市巴州区回风街道办事处龙舌坝村 4 组村民。该用地范围内的拆迁等补偿已于 2014 年由四川巴中经济开发区科技园管理委员会先行实施。 2016 年 7 月 13 日“英伦庄园”项目开工建设后，被告人李刚、李均等人向工程建设方提出承包挡土墙土石方开挖及工程基础旋挖桩开挖等工程，因没有施工资质，项目开发方未予应允。 2016 年 7 月 20 日至 8 月 12 日期间，被告人李刚、李均召集被告人李洪林、李建、李玉平等人在“四通茶楼”、“东某梦”茶楼商议采取阻挠施工的方式取得承包部分工程，并筹集了 7 万元阻工经费。同时被告人李均、李洪林、李建等人利用建立的微信群及打电话等方式邀约、发动本村村民和自己家人等 20 余人以索要土地两费为由，采取锁工地大门、拉电闸、辱骂、站在施工机械上等方式多次阻挠“英伦庄园”建筑工地施工，致使工地施工无法正常进行。经巴州区价格认证中心估价鉴定，挡工期间造成经济损失为 124530 元。"""},
    {"role": "assistant", "content": """某公司通过竞拍取得建设用地使用权，且用地范围内拆迁等补偿已经实施，但四组村民商议采取阻挠施工方式，筹集 7 万元阻工经费，发动 20 余人多次阻挠工地施工，经鉴定，挡工期间造成经济损失为 124,530 元。"""},
    {"role": "user", "content": """2018 年 1 月 15 日 14 时 10 分许，被告人莫新国酒后驾驶湘 A××××× 号小型轿车沿长沙市天心区伊莱克斯大道由南往北行驶至水电八局基地路段时被在该处执勤的长沙市公安局交通警察支队民警检查，经现场酒精吹气检测，测试结果显示其血液中乙醇含量为 195 毫克／ 100 毫升，随即被告人莫新国被交警带至湖南省融城医院抽取血样，并将血样送至长沙市公安局物证鉴定所检验，经检验，其血液中乙醇含量为 201.1 毫克／100 毫升。 2009 年 11 月 15 日，被告人莫新国经长沙市残疾人联合会审核为精神残疾人。 2018 年 5 月 28 日，经湖南省芙蓉司法鉴定中心鉴定，被告人莫新国作案时处于普通醉酒状态，实施危害行为时有完全刑事责任能力。 2018 年 1 月 30 日，被告人莫新国主动到公安机关投案，其归案后如实供述了自己的罪行。"""},
    {"role": "assistant", "content": """某人酒后驾驶车辆被交通警察检查，血液酒精含量 195 毫克／ 100 毫升。他随后被带到医院提取血样，检验结果显示酒精含量 201.1 毫克／ 100 毫升。他被认定为精神残疾人，但在案发时被鉴定为处于普通醉酒状态，有完全刑事责任能力。他在日后主动投案，如实供述罪行。"""}
]

import torch
from transformers.generation.utils import GenerationConfig


def get_model_response(prompt):
    # your function here 
    return None

def send_message(message: str):
    model_input = prompt + [{"role": "user", "content": message + """\n出于隐私保护目的，一定不能出现原先案件中的人名，机构名，时间，地点，金额，酒精含量，偷窃物品等具体信息！"""}]
    response = get_model_response(model_input)
    print(response)
    return response


TOTAL_AJ = 255_000

# NOTE you can run this on different sever
NODE = 6
NODE_ID = 6 # TODO
MAX_AJ_FOR_EACH_DIR = TOTAL_AJ / (2 * NODE)
SAVE_STEP = 100

def get_data(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data

def is_valid(aj_dict, MAX_LENGTH, MIN_LENGTH, MAX_NUM_FOR_EACH_AJAY, result, AJAY_number):
    if "uniqid" not in aj_dict or "AJDY" not in aj_dict:
        return False
    if "TITLE" not in aj_dict or "判决书" not in aj_dict["TITLE"]:
        return False
    if "AJAY" not in aj_dict or len(aj_dict["AJAY"]) == 0:
        return False
    if aj_dict["AJAY"][0] in AJAY_number and AJAY_number[aj_dict["AJAY"][0]] >= MAX_NUM_FOR_EACH_AJAY:
        return False
    if "SS" not in aj_dict or len(aj_dict["SS"]) > MAX_LENGTH or len(aj_dict["SS"]) < MIN_LENGTH:
        return False
    ss_list = [_dict["SS"] for _dict in result]
    if aj_dict["SS"] in ss_list:
        return False
    return True

def get_data_from_dir(dir_path):
    ms_or_xs = "ms" if "ms" in dir_path else "xs"
    MAX_NUMBER = 23273 if "ms" in dir_path else 6077
    MAX_LENGTH = 1600 if "ms" in dir_path else 900
    MIN_LENGTH = 600 if "ms" in dir_path else 300
    MAX_NUM_FOR_EACH_AJAY = 10_000 / NODE if "ms" in dir_path else 20_000 / NODE
    result = get_data(f"data_new_prompt_{ms_or_xs}_{NODE_ID}.json")
    AJAY_number = {}
    for _dict in result:
        if _dict["AJAY"][0] not in AJAY_number:
            AJAY_number[_dict["AJAY"][0]] = 1
        else:
            AJAY_number[_dict["AJAY"][0]] += 1
    for key, value in AJAY_number.items():
        if value >= MAX_NUM_FOR_EACH_AJAY:
            print(f"error in {ms_or_xs}, {key}, {value}")
    i_list = []
    for i in range(NODE_ID, MAX_NUMBER, NODE):
        i_list.append(i)
    i_list.reverse()
    for i in i_list:
        path = dir_path + str(i) + '.pb2.json'
        try:
            data = get_data(path)
        except:
            continue
        for aj_dict in data:
            if len(result) % SAVE_STEP == 0:
                with open(f"data_new_prompt_{ms_or_xs}_{NODE_ID}.json", 'w') as f:
                    json.dump(result, f, ensure_ascii=False, indent=4)
            if len(result) >= MAX_AJ_FOR_EACH_DIR:
                with open(f"data_new_prompt_{ms_or_xs}_{NODE_ID}.json", 'w') as f:
                    json.dump(result, f, ensure_ascii=False, indent=4)
                print(f"Processed {len(result)} samples in {dir_path}")
                return
            if not is_valid(aj_dict, MAX_LENGTH, MIN_LENGTH, MAX_NUM_FOR_EACH_AJAY, result, AJAY_number):
                continue
            query_key = make_query_by_rule_LLM(aj_dict["SS"])
            if query_key is None:
                print(f"no response")
                continue
            query, key = query_key
            processed_dict = {
                "uniqid": aj_dict["uniqid"],
                "title": aj_dict["TITLE"],
                "SS": aj_dict["SS"],
                "query": query,
                "key": key,
                "AJAY": aj_dict["AJAY"],
                "AJDY": aj_dict["AJDY"],
                "model": "chatglm3-6b"
            }
            result.append(processed_dict)
            print(f"processed {len(result)} in {dir_path}")
    with open(f"data_new_prompt_{ms_or_xs}_{NODE_ID}.json", 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"Processed {len(result)} samples in {dir_path}")

def make_query_by_rule_LLM(SS: str):
    # remove sentences after "the above fact" (上述事实) in the case description
    SS_list = SS.split("。")
    for i in range(len(SS_list)):
        if "上述事实" in SS_list[i]:
            SS_list = SS_list[:i]
            break
    SS = "。".join(SS_list)
    if len(SS) < 100:
        return None
    response = send_message(SS)
    if response is None:
        return None
    return response, SS

# NOTE raw legal case data here, you can download chinese crim
get_data_from_dir("xs/")