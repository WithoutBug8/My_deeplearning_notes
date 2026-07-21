"""
基于Steamlit完成WEB网页上传服务

核心机制： 当Web页面元素发生变化，则代码会重新执行一遍，可能会导致程序状态的丢失
"""

import time
import streamlit as st
from streamlit import session_state
from  knowledge_base import KnowledgeBaseService

# 添加网页标题
st.title("知识库更新服务")

# 添加文件上传服务
uploader_file = st.file_uploader(
    "请上传txt文件",
    type=['txt'],
    accept_multiple_files=False     # 仅接受一个文件上传
)
service = KnowledgeBaseService()
# 保存会话状态记录器,session_state是个字典
if "service" not in session_state:
    st.session_state["service"] = KnowledgeBaseService()


if uploader_file is not None:
    # 如果上传的文件内容不空，提取文件信息
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size / 1024   # 单位是KB

    # 在前端显示上传的文件内容
    st.subheader(f"文件名：{file_name}")
    st.write(f"格式：{file_type} | 大小：{file_size:.2f} KB")

    # 获取上传文件的内容,输出的格式是bytes字节数组，转换为字符串
    text = uploader_file.getvalue().decode('utf-8')

    # 把拿到的文件和文件名传递给处理函数
    with st.spinner("载入知识库中......"):
        time.sleep(1)
        result = st.session_state["service"].upload_by_str(text,file_name)
        st.write(result)
