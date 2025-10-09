# %%
import streamlit as st
import pandas as pd

st.title("我的第一个 Streamlit 应用")
st.header("这是一个二级标题")
st.write("你好，世界！Streamlit 让创建 Web 应用变得非常简单。")
st.markdown("我们可以使用 **Markdown** 来格式化文本。")

st.subheader("显示一个 Pandas DataFrame")
df = pd.DataFrame({
    '第一列': [1, 2, 3, 4],
    '第二列': [10, 20, 30, 40]
})
st.write("这是一个 DataFrame:")
st.dataframe(df)

st.subheader("添加一个按钮")
if st.button("点我一下"):
    # 当按钮被点击时，这部分代码会执行
    st.write("你点击了按钮！🎉")
else:
    st.write("请点击上面的按钮。")
