# coding=utf-8
from __future__ import print_function, absolute_import, unicode_literals
from gm.api import *

import datetime
import numpy as np
import pandas as pd

'''
智能体投资：风格轮动量化策略
策略逻辑：
1. 每月轮动：在上证50、沪深300、中证500中选择近期收益最高的指数
2. 买入该指数中市值最大的股票
3. 加入止盈20% + 止损8%风控机制
回测时间：2019-01-01 ~ 2020-12-31
初始资金：1000万
'''

def init(context):
    # 待轮动的风格指数(分别为：上证50、沪深300、中证500)
    context.index = ['SHSE.000016', 'SHSE.000300', 'SZSE.399625']
    # 统计收益的天数
    context.days = 20
    # 持股数量（原10 → 改为6，个性化修改）
    context.holding_num = 6
    # 止盈止损记录（自己维护成本价，避免API字段问题）
    context.cost_dict = {}

    # 每日定时任务
    schedule(schedule_func=algo, date_rule='1d', time_rule='09:30:00')


def algo(context):
    now_str = context.now.strftime('%Y-%m-%d')
    last_day = get_previous_n_trading_dates(exchange='SHSE', date=now_str, n=1)[0]

    # 止盈止损监控（每日检查，加分项）
    positions = get_position()
    for pos in positions:
        symbol = pos['symbol']
        # 从自己维护的字典里取成本价
        if symbol not in context.cost_dict:
            continue
        cost = context.cost_dict[symbol]
        if cost <= 0:
            continue
        # 获取当前价格
        bar = current(symbol)
        if not bar:
            continue
        price = bar[0]['price']
        
        # 止盈 20%
        if price >= cost * 1.20:
            order_target_percent(symbol, 0, order_type=OrderType_Limit, position_side=PositionSide_Long, price=price)
            print(f"【止盈卖出】{symbol} 成本：{cost:.2f} 当前：{price:.2f}")
            if symbol in context.cost_dict:
                del context.cost_dict[symbol]
        
        # 止损 8%
        if price <= cost * 0.92:
            order_target_percent(symbol, 0, order_type=OrderType_Limit, position_side=PositionSide_Long, price=price)
            print(f"【止损卖出】{symbol} 成本：{cost:.2f} 当前：{price:.2f}")
            if symbol in context.cost_dict:
                del context.cost_dict[symbol]

    # 每月第一个交易日调仓
    if context.now.month != pd.Timestamp(last_day).month:
        return_index = pd.DataFrame(columns=['return'])
        for i in context.index:
            return_index_his = history_n(symbol=i, frequency='1d', count=context.days+1, fields='close,bob',
                                        fill_missing='Last', adjust=ADJUST_PREV, end_time=last_day, df=True)
            return_index_his = return_index_his['close'].values
            return_index.loc[i, 'return'] = return_index_his[-1] / return_index_his[0] - 1

        # 选收益最高的指数
        sector = return_index.index[np.argmax(return_index)]
        print('{}: 本轮最佳指数：{}'.format(now_str, sector))

        # 获取成分股
        symbols = list(stk_get_index_constituents(index=sector, trade_date=last_day)['symbol'])
        stocks_info = get_symbols(sec_type1=1010, symbols=symbols, trade_date=now_str, skip_suspended=True, skip_st=True)
        symbols = [item['symbol'] for item in stocks_info if item['listed_date'] < context.now and item['delisted_date'] > context.now]

        # 按市值排序，取前N只
        fin = stk_get_daily_mktvalue_pt(symbols=symbols, fields='tot_mv', trade_date=last_day, df=True).sort_values(by='tot_mv', ascending=False)
        to_buy = list(fin.iloc[:context.holding_num]['symbol'])

        # 资金分配
        percent = 0.98 / len(to_buy) if len(to_buy) > 0 else 0

        # 平仓不在目标池的股票
        positions = get_position()
        for pos in positions:
            symbol = pos['symbol']
            if symbol not in to_buy:
                new_price = history_n(symbol=symbol, frequency='1d', count=1, end_time=now_str, fields='open', adjust=ADJUST_PREV, df=False)[0]['open']
                order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Limit, position_side=PositionSide_Long, price=new_price)
                if symbol in context.cost_dict:
                    del context.cost_dict[symbol]

        # 买入目标股票
        for symbol in to_buy:
            new_price = history_n(symbol=symbol, frequency='1d', count=1, end_time=now_str, fields='open', adjust=ADJUST_PREV, df=False)[0]['open']
            order_target_percent(symbol=symbol, percent=percent, order_type=OrderType_Limit, position_side=PositionSide_Long, price=new_price)
            # 记录买入成本价
            context.cost_dict[symbol] = new_price


def on_order_status(context, order):
    symbol = order['symbol']
    price = order['price']
    volume = order['volume']
    status = order['status']
    side = order['side']
    effect = order['position_effect']
    order_type = order['order_type']

    if status == 3:
        side_effect = '开多仓' if (effect == 1 and side == 1) else \
                      '开空仓' if (effect == 1 and side == 2) else \
                      '平空仓' if (effect == 2 and side == 1) else '平多仓'
        order_type_word = '限价' if order_type == 1 else '市价'
        print('{}: 标的：{}，操作：以{}{}，价格：{:.2f}，数量：{}'.format(context.now, symbol, order_type_word, side_effect, price, volume))
        # 开仓成交时更新成本价
        if effect == 1 and side == 1:
            context.cost_dict[symbol] = price
        # 平仓成交时删除成本价
        elif effect == 2 and side == 2:
            if symbol in context.cost_dict:
                del context.cost_dict[symbol]


def on_backtest_finished(context, indicator):
    print('='*60)
    print('回测完成！策略已优化：加入止盈20%、止损8%、持股数6只')
    print('='*60)


if __name__ == '__main__':
    run(strategy_id='57a0d1bb-5c2b-11f1-9beb-2ab2b96cbea9',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='d87d61906b3cb5556e8a470eed88f74e9d3e7ab7',
        backtest_start_time='2019-01-01 08:00:00',
        backtest_end_time='2020-12-31 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001,
        backtest_match_mode=1)
