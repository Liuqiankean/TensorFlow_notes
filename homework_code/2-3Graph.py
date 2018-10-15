
# coding: utf-8

# In[36]:


#2.图与会话
import tensorflow as tf
c=tf.add(3,5)
with tf.Session()as sess1:
    res=sess1.run(c)
sess2=tf.Session()
print(c.eval())


# In[26]:


b=tf.convert_to_tensor([[1,2,3],[1,2,3]])
c=tf.constant([[1,2,3],[1,2,3]])

with tf.Session() as sess:
    print(sess.run(tf.ones(c.shape)))


# In[32]:


#3.图的边与节点作业
a.[1,:,:,:]#利用索引取出第二张图片
tf.slice(a,[1,0,0,0],[1,28,28,3])#利用切片取出第2张图片
#索引是从点开始取，切片是片段

#取出其中的第1、3、5、7张图片
i=0
while i<=7:
    tf.slice(a,[i,0,0,0],[1,28,28,3])
    i+=2

#取出第6-8张（包括6不包括8）图片中中心区域（14*14）的部分
for i in rang(6,8):
    tf.slice(a,[i,13,13,0],[1,14,14,3])

#将图片根据通道拆分成三份单通道图片
for i in rang(0,10):
    for j in rang(0,3):
        tf.slice(a,[i,0,0,0],[1,28,28,j])

#`tf.shape(img)`返回的张量的阶数=4
#`shape`属性的值(1,4)


# In[ ]:




