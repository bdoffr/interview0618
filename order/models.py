from django.db import models
from django.contrib.auth.models import User

class Product(models.Model):
    """商品表"""
    name = models.CharField(max_length=200, verbose_name="商品名称")
    description = models.TextField(verbose_name="商品描述")
    price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="价格")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

class Stock(models.Model):
    """库存表 (与商品一对一)"""
    product = models.OneToOneField(Product, on_delete=models.CASCADE, primary_key=True)
    quantity = models.PositiveIntegerField(default=0, verbose_name="库存数量")
    # 乐观锁的版本号
    version = models.PositiveIntegerField(default=0)

    def __str__(self):
        return f"{self.product.name} - 库存: {self.quantity}"

class Order(models.Model):
    """订单主表"""
    STATUS_CHOICES = [
        ('PENDING', '待处理'),
        ('PROCESSING', '处理中'),
        ('COMPLETED', '已完成'),
        ('FAILED', '失败'),
    ]
    user = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name="用户")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    total_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    created_at = models.DateTimeField(auto_now_add=True)

class OrderDetail(models.Model):
    """订单明细表"""
    order = models.ForeignKey(Order, related_name='details', on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.PROTECT)
    quantity = models.PositiveIntegerField()
    price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="下单时价格")