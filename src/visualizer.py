import json
import networkx as nx
from pyvis.network import Network
from collections import defaultdict
import os

def build_character_graph_optimized(entities_path="outputs/entities.json", 
                                  output_html="outputs/character_graph.html"):
    """
    构建人物关系图（优化版）：
    - 使用更高效的数据结构减少计算复杂度
    - 优化图的构建和渲染过程
    """
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_html)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 读取抽取结果
        with open(entities_path, "r", encoding="utf-8") as f:
            chapters = json.load(f)

        G = nx.Graph()
        
        # 先收集所有章节中的人物列表
        chapter_characters = []
        for chap in chapters:
            if "entities" not in chap or not chap["entities"]:
                continue
            
            characters = []
            for e in chap["entities"]:
                if isinstance(e, dict) and "characters" in e:
                    for char in e["characters"]:
                        if isinstance(char, dict) and "name" in char:
                            characters.append(char["name"])
            
            if characters:
                chapter_characters.append(characters)
        
        # 统计人物共现次数
        co_occurrence = defaultdict(int)
        for chars in chapter_characters:
            # 去重，避免同一章节中同一个人的多次出现导致权重计算错误
            unique_chars = list(set(chars))
            
            # 同章人物两两连边
            for i, c1 in enumerate(unique_chars):
                for c2 in unique_chars[i+1:]:
                    # 确保顺序一致，避免(c1,c2)和(c2,c1)被视为不同的边
                    if c1 > c2:
                        c1, c2 = c2, c1
                    co_occurrence[(c1, c2)] += 1
        
        # 构建图
        for (c1, c2), weight in co_occurrence.items():
            G.add_edge(c1, c2, weight=weight)
        
        # 过滤低权重边，减少图的复杂度
        min_weight = 2  # 只保留出现至少2次的关系
        edges_to_keep = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] >= min_weight]
        G = G.edge_subgraph(edges_to_keep)
        
        # 只保留有边连接的节点
        G = G.subgraph([n for n in G.nodes if G.degree(n) > 0])
        
        # 使用 PyVis 生成可交互网页
        # 修改参数，解决渲染问题
        net = Network(height="1000px", width="100%", notebook=False, directed=False)
        net.barnes_hut()  # 使用更高效的布局算法
        
        # 添加节点和边
        for node in G.nodes:
            net.add_node(node, label=node)
        for u, v, data in G.edges(data=True):
            net.add_edge(u, v, value=data["weight"])
        
        # 配置物理参数以提高渲染性能
        net.set_options("""
        var options = {
          "nodes": {
            "font": {
              "size": 14
            },
            "scaling": {
              "min": 20,
              "max": 30
            }
          },
          "edges": {
            "scaling": {
              "min": 1,
              "max": 5,
              "label": {
                "enabled": false
              }
            },
            "smooth": {
              "forceDirection": "none",
              "roundness": 0.4
            }
          },
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springConstant": 0.001,
              "springLength": 200,
              "damping": 0.09,
              "avoidOverlap": 0
            },
            "minVelocity": 0.75
          }
        }""")
        
        # 修改HTML生成方式，避免使用show()方法的问题
        net.write_html(output_html, open_browser=False, notebook=False)
        print(f"✅ 优化后的人物关系图已生成: {os.path.abspath(output_html)}")
        
    except Exception as e:
        print(f"❌ 生成人物关系图时出错: {e}")
        # 提供更详细的错误信息和解决建议
        import traceback
        traceback.print_exc()
        print("\n建议尝试以下解决方法：")
        print("1. 确保PyVis库版本正确，可以尝试更新：pip install pyvis --upgrade")
        print("2. 检查outputs目录是否存在且可写")
        print("3. 检查entities.json文件格式是否正确")

# 添加初始化执行方法，使文件可以直接运行
if __name__ == "__main__":
    # 调用优化版的人物关系图构建函数
    build_character_graph_optimized()