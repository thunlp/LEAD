import requests
import argparse
from tqdm import tqdm
import json
parser = argparse.ArgumentParser(description="Help info.")
parser.add_argument('--q', type=str, default='data/query/query.json', help='Query path.')
parser.add_argument('--old', type=str, default='data/query/query_old.json', help='Query path.')
parser.add_argument('--w', type=str, default='data/query/query_simplify.json', help='Write path.')

args = parser.parse_args()

prefix = [
    {
      "role": "system",
      "content": "你是一个法律专家，能够从案件基本情况中提取关键要素。"
    },
    {
      "role": "user",
      "content": "我现在有一个数据集, 该数据集是用来以案搜案的, 具体来说, 给定案件基本情况, 我需要用它去数据库里寻找与之最相似的其他案件, 现在有个问题, 我的数据集里面的案件基本情况都太长了, 实际上普通人想要以案搜案时, 给出的案件基本情况可能都非常简短, 只包含一些关键性的基本事实, 与现有的数据集不同. 具体而言, 现在案件基本情况的平均token数量为700左右, 但我想构造一个新的精简的数据集, 让它的平均token数量缩短至100左右, 我会把这些案件基本情况都发给你, 请你帮我简化它们, 使其长度大大缩短,使其长度大大缩短，例如你可以去掉案件基本情况中并不关键的人名，地点，精简语言等，并去掉“被告人”等诉讼时用语，但不可以删掉对于案件判决有重要作用的事实，如伤情判定，是否取得谅解，金额数目等等。除了精简之后的案件基本情况之外不要有任何其他的内容"
    },
    {
      "role": "assistant",
      "content": "理解您的需求，您希望将提供的案件基本情况简化为大约100个token左右，去除不必要的细节和信息。\n\n请提供更多的案件基本情况，以便我可以帮您进一步精简它们。"
    },
    {
      "role": "user",
      "content": "2015年8月，巴中鑫金地实业有限公司通过公开竞拍的方式，取得了巴州区回风津桥湖片区国有建设用地使用权，开发建设“英伦庄园”项目。被告人李刚、李均、李洪林、李建、李玉平系该项目用地范围内的巴中市巴州区回风街道办事处龙舌坝村4组村民。该用地范围内的拆迁等补偿已于2014年由四川巴中经济开发区科技园管理委员会先行实施。2016年7月13日“英伦庄园”项目开工建设后，被告人李刚、李均等人向工程建设方提出承包挡土墙土石方开挖及工程基础旋挖桩开挖等工程，因没有施工资质，项目开发方未予应允。2016年7月20日至8月12日期间，被告人李刚、李均召集被告人李洪林、李建、李玉平等人在“四通茶楼”、“东某梦”茶楼商议采取阻挠施工的方式取得承包部分工程，并筹集了7万元阻工经费。同时被告人李均、李洪林、李建等人利用建立的微信群及打电话等方式邀约、发动本村村民和自己家人等20余人以索要土地两费为由，采取锁工地大门、拉电闸、辱骂、站在施工机械上等方式多次阻挠“英伦庄园”建筑工地施工，致使工地施工无法正常进行。经巴州区价格认证中心估价鉴定，挡工期间造成经济损失为124530元。"
    },
    {
      "role": "assistant",
      "content": "巴中鑫金地实业有限公司通过竞拍取得建设用地使用权，且用地范围内拆迁等补偿已经实施，但四组村民商议采取阻挠施工方式，筹集7万元阻工经费，发动20余人多次阻挠工地施工，经鉴定，挡工期间造成经济损失为124,530元。"
    },
    {
      "role": "user",
      "content": "2018年1月15日14时10分许，被告人莫新国酒后驾驶湘A×××××号小型轿车沿长沙市天心区伊莱克斯大道由南往北行驶至水电八局基地路段时被在该处执勤的长沙市公安局交通警察支队民警检查，经现场酒精吹气检测，测试结果显示其血液中乙醇含量为195毫克／100毫升，随即被告人莫新国被交警带至湖南省融城医院抽取血样，并将血样送至长沙市公安局物证鉴定所检验，经检验，其血液中乙醇含量为201.1毫克／100毫升。2009年11月15日，被告人莫新国经长沙市残疾人联合会审核为精神残疾人。2018年5月28日，经湖南省芙蓉司法鉴定中心鉴定，被告人莫新国作案时处于普通醉酒状态，实施危害行为时有完全刑事责任能力。2018年1月30日，被告人莫新国主动到公安机关投案，其归案后如实供述了自己的罪行。"
    },
    {
      "role": "assistant",
      "content": "莫新国在长沙市酒后驾驶车辆被交通警察检查，血液酒精含量195毫克／100毫升。他随后被带到医院提取血样，检验结果显示酒精含量201.1毫克／100毫升。被告人莫新国被认定为精神残疾人，但在案发时被鉴定为处于普通醉酒状态，有完全刑事责任能力。他在日后主动投案，如实供述罪行。"
    }
]

with open(args.q, 'r') as f:
    lines = f.readlines()

with open(args.old, 'r') as f:
    query_old = json.load(f)

query_simplify = []
for line in tqdm(lines[:]):
    query_dict = eval(line)
    # if query_dict["ridx"] == 5156 or query_dict["ridx"] == 4891:
    #     continue
    # if query_dict["ridx"] == 5187: # test TODO
    #     continue
    ridx = query_dict["ridx"]
    find = False
    for i in range(len(query_old)):
        if query_old[i]["ridx"] == ridx:
            if query_old[i]["q_short"] != "error":
                print(f"{ridx} is ok")
                query_simplify.append(query_old[i])
                find = True
            break
    if find:
        continue
    long_query = query_dict['q'].strip('"')
    role_content = {
        "role": "user",
        "content": long_query
    }
    response_json = requests.post("http://120.92.10.46:8080/chat", json={
        "messages": prefix + [role_content],
        "temperature": 0.2,
        "top_p": 1,
        "max_tokens": 512,
        "presence_penalty": 0,
        "frequency_penalty": 0
    }).json()
    if response_json == {'error': 'no account available'}:
        print(f"error in {query_dict['ridx']}")
        q_short = "error"
        tokens = 0
    else:
        q_short = response_json["choices"][0]["message"]["content"]
        tokens = response_json["usage"]["completion_tokens"]
    # print(response_json)
    query_simplify.append({"path": query_dict["path"], 
                           "ridx": query_dict["ridx"], 
                           "q": query_dict["q"], 
                           "q_short": q_short,
                           "tokens": tokens,
                           "crime": query_dict["crime"]})


with open(args.w, 'w') as f:
    json.dump(query_simplify, f, ensure_ascii=False)

# response_json = requests.post("http://47.254.22.102:8989/chat", json={
#     "model": "gpt-3.5-turbo-0613",
#         "messages": [{
#           "role": "user",
#           "content": "王X酒后无证驾驶摩托车，未接受民警检查并辱骂民警。王X1和李X见状后一起辱骂并殴打民警阮X、田X、陈X。经鉴定，田X、阮X、陈X受到轻微伤。王X血液中乙醇含量为106mg／100ml。"
# }],
#         "temperature": 0.2,
#         "top_p": 1,
#         "max_tokens": 512,
#         "presence_penalty": 0,
#         "frequency_penalty": 0
#     }).json()

# print(response_json)