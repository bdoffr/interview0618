# 项目设计说明
## models
- Product和Stock分离：库存是变化非常频繁的数据,将其分离可以减少对商品主表的锁竞争和写操作
- Stock.version：实现乐观锁预留版本号字段
- OrderDetail.price：记录下单时的价格快照,防止商品价格变动影响历史订单
## views
### 批量订单处理
- 在一个请求中处理多个商品下单,保证数据一致性,使用悲观锁select_for_update来确保原子性
- transaction.atomic():保证每个商品 检查库存 -> 扣减库存 -> 创建明细 这个过程的原子性
- select_for_update():悲观锁防止超卖
### 商品搜索功能
### 缓存策略
- 商品信息:Hash结构存储
  - Key: product:{product_id}
  - Value: {'name': 'xxx', 'price': 'yyy'}
- 库存信息:键值对
  - Key: stock:{product_id}
  - Value: 库存数量
- 搜索结果:缓存搜索结果的商品ID列表
  - Key:search:{query_hash}(搜索关键词做哈希)
  - Value:[1, 5, 12] (商品 ID 列表)
  - TTL:设置过期时间