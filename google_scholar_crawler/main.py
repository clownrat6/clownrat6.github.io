import os
import json
import time
import random
import traceback
from datetime import datetime

from scholarly import scholarly, MaxTriesExceededException





try:
    # 配置scholarly
    scholarly.set_retries(3)
    scholarly.set_timeout(10)  # 10秒超时
    
    # 打印进度
    print(f"开始查询学者ID: {os.environ['GOOGLE_SCHOLAR_ID']} {os.environ['GOOGLE_SCHOLAR_ID'] is not None}")
    start_time = time.time()

    # 查询作者信息
    author = scholarly.search_author_id(os.environ['GOOGLE_SCHOLAR_ID'])
    
    # 检查是否找到作者
    if author is None:
        raise ValueError(f"未找到学者ID: {os.environ['GOOGLE_SCHOLAR_ID']}")
    
    print(f"✅ 获取学者基本信息成功，耗时 {time.time()-start_time:.2f} 秒")
    
    # 随机延迟，模拟人类行为
    time.sleep(random.uniform(2, 5))
    
    # 填充详细数据
    print("开始填充详细数据...")
    scholarly.fill(author, sections=['basics', 'indices', 'counts', 'publications'])
    print(f"✅ 填充详细数据成功，耗时 {time.time()-start_time:.2f} 秒")
    
    # 处理数据
    name = author['name']
    author['updated'] = str(datetime.now())
    author['publications'] = {v['author_pub_id']: v for v in author['publications']}
    
    # 保存数据
    os.makedirs('results', exist_ok=True)
    with open(f'results/gs_data.json', 'w') as outfile:
        json.dump(author, outfile, ensure_ascii=False, indent=2)
    
    # 生成Shields.io数据
    shieldio_data = {
        "schemaVersion": 1,
        "label": "citations",
        "message": f"{author['citedby']}",
    }
    
    with open(f'results/gs_data_shieldsio.json', 'w') as outfile:
        json.dump(shieldio_data, outfile, ensure_ascii=False)
    
    print(f"✅ 所有数据已保存到 results 目录")

except MaxTriesExceededException:
    print("❌ 错误: 达到最大重试次数，可能被Google Scholar封禁")
    exit(1)
except:
    traceback.print_exc()
# except ValueError as ve:
#     print(f"❌ 错误: {str(ve)}")
#     exit(1)
# except Exception as e:
#     print(f"❌ 错误: {str(e)}")
#     exit(1)
