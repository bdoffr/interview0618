from django.db import transaction
from .models import Product, Stock, Order, OrderDetail


class InsufficientStockError(Exception):
    """自定义库存不足异常"""
    pass


class OrderService:
    def __init__(self, user):
        self.user = user
        self.order = None

    def process_batch_order(self, order_items):
        results = []

        # 创建一个主订单，状态为 PENDING
        self.order = Order.objects.create(user=self.user, status='PENDING')

        # 处理每个商品
        for item in order_items:
            product_id = item.get('product_id')
            quantity = item.get('quantity')

            try:
                # 每个商品的扣减都是一个独立的原子操作
                with transaction.atomic():
                    # 使用悲观锁锁定库存行，防止并发修改
                    stock = Stock.objects.select_for_update().get(product_id=product_id)

                    if stock.quantity < quantity:
                        raise InsufficientStockError(f"商品 {product_id} 库存不足")

                    # 扣减库存
                    stock.quantity -= quantity
                    stock.save()

                    # 获取商品信息用于创建订单详情
                    product = stock.product

                    # 创建订单明细
                    OrderDetail.objects.create(
                        order=self.order,
                        product=product,
                        quantity=quantity,
                        price=product.price  # 记录价格快照
                    )

                    results.append({"product_id": product_id, "success": True})

            except Stock.DoesNotExist:
                results.append({"product_id": product_id, "success": False, "error": "商品不存在"})
            except InsufficientStockError as e:
                results.append({"product_id": product_id, "success": False, "error": str(e)})
            except Exception as e:
                # 记录未知错误日志
                results.append({"product_id": product_id, "success": False, "error": "未知系统错误"})

        # 计算总金额并更新订单状态
        if any(res['success'] for res in results):
            total_amount = sum(d.price * d.quantity for d in self.order.details.all())
            self.order.total_amount = total_amount
            self.order.status = 'PROCESSING'  # 或 'COMPLETED'，取决于业务流程
            self.order.save()
        else:
            # 如果所有商品都失败了，可以删除这个空订单
            self.order.delete()
            self.order = None

        return results
