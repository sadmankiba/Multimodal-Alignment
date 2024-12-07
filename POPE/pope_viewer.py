import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

df = pd.read_parquet("test.parquet", engine="pyarrow")
print(len(df))

for idx, row in df.iterrows():
    # print(row)
    image = np.array(Image.open(io.BytesIO(row["image"]["bytes"])))
    question = row["question"]
    answer = row["answer"]
    print(question)
    print(answer)
    # plt.imshow(image)
    # plt.show()
    break
