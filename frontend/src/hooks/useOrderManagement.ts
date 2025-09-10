/*!
 * Advanced Order Management Hook with Real-Time Updates
 * Handles order lifecycle, real-time status updates, and position tracking
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useWebSocket } from './useWebSocket';

interface Order {
  id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  orderType: 'MARKET' | 'LIMIT' | 'STOP' | 'STOP_LIMIT';
  quantity: number;
  price?: number;
  stopPrice?: number;
  timeInForce: 'DAY' | 'GTC' | 'IOC' | 'FOK';
  status: 'PENDING' | 'SUBMITTED' | 'PARTIAL' | 'FILLED' | 'CANCELLED' | 'REJECTED';
  filledQuantity: number;
  averagePrice: number;
  timestamp: number;
  strategy?: string;
}

interface OrderUpdate {
  orderId: string;
  status: string;
  filledQuantity: number;
  averagePrice: number;
  timestamp: number;
}

export const useOrderManagement = () => {
  const [orders, setOrders] = useState<Map<string, Order>>(new Map());
  const [positions, setPositions] = useState<Map<string, number>>(new Map());
  const [orderHistory, setOrderHistory] = useState<Order[]>([]);
  const { subscribe, unsubscribe, sendMessage } = useWebSocket();
  
  const ordersRef = useRef(orders);
  ordersRef.current = orders;

  // Subscribe to order updates
  useEffect(() => {
    const handleOrderUpdate = (update: OrderUpdate) => {
      setOrders(prev => {
        const newOrders = new Map(prev);
        const existingOrder = newOrders.get(update.orderId);
        
        if (existingOrder) {
          const updatedOrder: Order = {
            ...existingOrder,
            status: update.status as Order['status'],
            filledQuantity: update.filledQuantity,
            averagePrice: update.averagePrice,
            timestamp: update.timestamp
          };
          
          newOrders.set(update.orderId, updatedOrder);
          
          // Update positions if order is filled
          if (updatedOrder.status === 'FILLED' || updatedOrder.status === 'PARTIAL') {
            updatePosition(updatedOrder.symbol, updatedOrder.side, update.filledQuantity - existingOrder.filledQuantity);
          }
          
          // Move to history if complete
          if (updatedOrder.status === 'FILLED' || updatedOrder.status === 'CANCELLED' || updatedOrder.status === 'REJECTED') {
            setOrderHistory(prev => [updatedOrder, ...prev.slice(0, 999)]);
          }
        }
        
        return newOrders;
      });
    };

    subscribe('ORDER_UPDATE', handleOrderUpdate);
    return () => unsubscribe('ORDER_UPDATE', handleOrderUpdate);
  }, [subscribe, unsubscribe]);

  const updatePosition = useCallback((symbol: string, side: 'BUY' | 'SELL', quantity: number) => {
    setPositions(prev => {
      const newPositions = new Map(prev);
      const currentPosition = newPositions.get(symbol) || 0;
      const adjustment = side === 'BUY' ? quantity : -quantity;
      const newPosition = currentPosition + adjustment;
      
      if (newPosition === 0) {
        newPositions.delete(symbol);
      } else {
        newPositions.set(symbol, newPosition);
      }
      
      return newPositions;
    });
  }, []);

  const submitOrder = useCallback(async (orderRequest: Omit<Order, 'id' | 'status' | 'filledQuantity' | 'averagePrice' | 'timestamp'>) => {
    const order: Order = {
      ...orderRequest,
      id: `order_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      status: 'PENDING',
      filledQuantity: 0,
      averagePrice: 0,
      timestamp: Date.now()
    };

    setOrders(prev => new Map(prev.set(order.id, order)));

    try {
      await sendMessage({
        type: 'SUBMIT_ORDER',
        order: {
          client_order_id: order.id,
          symbol: order.symbol,
          side: order.side,
          order_type: order.orderType,
          quantity: order.quantity,
          price: order.price,
          stop_price: order.stopPrice,
          time_in_force: order.timeInForce,
          strategy_id: order.strategy
        }
      });

      setOrders(prev => {
        const newOrders = new Map(prev);
        newOrders.set(order.id, { ...order, status: 'SUBMITTED' });
        return newOrders;
      });

      return order.id;
    } catch (error) {
      setOrders(prev => {
        const newOrders = new Map(prev);
        newOrders.set(order.id, { ...order, status: 'REJECTED' });
        return newOrders;
      });
      throw error;
    }
  }, [sendMessage]);

  const cancelOrder = useCallback(async (orderId: string) => {
    const order = orders.get(orderId);
    if (!order || ['FILLED', 'CANCELLED', 'REJECTED'].includes(order.status)) {
      throw new Error('Cannot cancel order');
    }

    try {
      await sendMessage({
        type: 'CANCEL_ORDER',
        order_id: orderId
      });
    } catch (error) {
      console.error('Failed to cancel order:', error);
      throw error;
    }
  }, [orders, sendMessage]);

  const modifyOrder = useCallback(async (orderId: string, modifications: Partial<Pick<Order, 'quantity' | 'price' | 'stopPrice'>>) => {
    const order = orders.get(orderId);
    if (!order || !['SUBMITTED', 'PARTIAL'].includes(order.status)) {
      throw new Error('Cannot modify order');
    }

    try {
      await sendMessage({
        type: 'MODIFY_ORDER',
        order_id: orderId,
        modifications
      });
    } catch (error) {
      console.error('Failed to modify order:', error);
      throw error;
    }
  }, [orders, sendMessage]);

  const getOpenOrders = useCallback(() => {
    return Array.from(orders.values()).filter(order => 
      ['PENDING', 'SUBMITTED', 'PARTIAL'].includes(order.status)
    );
  }, [orders]);

  const getOrdersBySymbol = useCallback((symbol: string) => {
    return Array.from(orders.values()).filter(order => order.symbol === symbol);
  }, [orders]);

  const getOrdersByStrategy = useCallback((strategy: string) => {
    return Array.from(orders.values()).filter(order => order.strategy === strategy);
  }, [orders]);

  return {
    orders: Array.from(orders.values()),
    positions: Array.from(positions.entries()),
    orderHistory,
    submitOrder,
    cancelOrder,
    modifyOrder,
    getOpenOrders,
    getOrdersBySymbol,
    getOrdersByStrategy
  };
};
