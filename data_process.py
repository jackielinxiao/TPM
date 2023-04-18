import pandas as pd
import numpy as np

lines = pd.read_csv("./data/full_matrix.csv")
lines.fillna("12345", inplace=True)

lines = lines[lines["duration"]<6000]

durations = lines["duration"]

percen_value = np.percentile(durations, np.linspace(0.0, 100.0, num=32+1).astype(np.float32)).tolist()

print (percen_value)

percen_value = np.percentile(durations, np.linspace(0.0, 100.0, num=99+1).astype(np.float32)).tolist()

print (percen_value)


print(lines.head(4))
print(len(lines))

alldata = []
for index, sp in lines.iterrows():
  if index % 10000 == 0:
    print(index)
  dur = sp["duration"]
  query = sp["searchstring.tokens"].replace("\"","")
  item_list = sp["itemlist"].replace("\"","")

  aaa = []
  qs = [int(i) for i in query.split(",")][:10]  # q
  qlen = len(qs)
  qs = qs+[654321]*(10-len(qs))
  aaa.extend(qs)

  ts = [int(i) for i in item_list.split(",")][:10]  # item
  tlen = len(ts)
  ts = ts+[654321]*(10-len(ts))
  aaa.extend(ts)

  qm = [1.0]*qlen +[0.0]*(10-qlen)
  aaa.extend(qm)

  tm = [1.0]*tlen +[0.0]*(10-tlen)
  aaa.extend(tm)

  aaa.append(float(dur)/1000.0)
  aaa.append(qlen)
  aaa.append(tlen)

  alldata.append(aaa)

alldata= np.array(alldata).astype(float)
np.random.shuffle(alldata)

train = alldata[:int(0.8*len(alldata))]
test = alldata[int(0.8*len(alldata)):]

print(train.shape)
print(test.shape)

print(train[0,:])

print(train[:,-3])
np.save("train_clip.npy", train)
np.save("test_clip.npy", test)
