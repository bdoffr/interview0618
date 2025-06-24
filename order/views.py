import logging

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .services import OrderService, InsufficientStockError

logger = logging.getLogger('django')


class BatchOrderAPIView(APIView):
    def post(self, request, *args, **kwargs):
        """
        批量下单接口
        请求体格式: [{"product_id": 1, "quantity": 1}, {"product_id": 2, "quantity": 2}]
        """
        order_items = request.data
        if not isinstance(order_items, list) or not order_items:
            return Response({"error": "无效的订单格式"}, status=status.HTTP_400_BAD_REQUEST)

        # 调用service层处理业务逻辑
        order_service = OrderService(user=request.user)
        logger.info(f"service: {order_service}")
        results = order_service.process_batch_order(order_items)
        logger.info(f"results: {results}")

        # 根据处理结果，判断整个订单的最终状态
        has_success = any(item['success'] for item in results)

        if not has_success:
            # 如果没有任何一个商品成功，则返回失败
            return Response({
                "message": "所有商品下单失败",
                "details": results
            }, status=status.HTTP_400_BAD_REQUEST)

        return Response({
            "message": "订单处理完成",
            "order_id": order_service.order.id if order_service.order else None,
            "details": results
        }, status=status.HTTP_201_CREATED)
