# API 服务
```shell
    python api_main.py
```

端口在 api_main.py 中修改port值.

多进程或单进程模式，可以在 api_main.py 末尾处修改：
```python
if __name__ == "__main__":
    # 单进程 + 协程
    # run(False)
    # 多进程 + 协程
    run(True)
```
