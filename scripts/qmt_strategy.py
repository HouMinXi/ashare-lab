#coding:utf-8
"""QMT strategy: monitor order files and execute trades.

Runs inside QMT strategy framework. Reads orders from
H:/ashare-lab/orders/{date}.json, executes via passorder(), writes
fills to H:/ashare-lab/fills/{date}.json.

Usage: Load this file in QMT UI -> Strategy -> New Strategy -> Run
"""

import json
import os
import time
import logging

log = logging.getLogger('qmt_strategy')

ACCOUNT_ID = '8890836643'
ORDER_DIR = r'H:\ashare-lab\orders'
FILL_DIR = r'H:\ashare-lab\fills'

# Price type constants (from xtconstant)
STOCK_BUY = 23
STOCK_SELL = 24
FIX_PRICE = 11
LATEST_PRICE = 5


def init(ContextInfo):
    """Initialize strategy: set account, enable callbacks."""
    ContextInfo.set_account(ACCOUNT_ID)
    ContextInfo.set_auto_trade_callback(1)

    # State tracking
    ContextInfo.submitted_orders = set()  # order_ids already submitted
    ContextInfo.fill_results = []  # collected fill results
    ContextInfo.current_date = None
    ContextInfo.order_file_path = None
    ContextInfo.fill_file_path = None

    log.info('QMT strategy initialized, account=%s', ACCOUNT_ID)


def handlebar(ContextInfo):
    """Main loop: check for order file and execute."""
    today = _get_today(ContextInfo)
    if today is None:
        return

    # Update file paths if date changed
    if today != ContextInfo.current_date:
        ContextInfo.current_date = today
        ContextInfo.order_file_path = os.path.join(
            ORDER_DIR, '{}.json'.format(today))
        ContextInfo.fill_file_path = os.path.join(
            FILL_DIR, '{}.json'.format(today))
        ContextInfo.submitted_orders = set()
        ContextInfo.fill_results = []
        log.info('New trading day: %s', today)

    # Check if order file exists
    if not os.path.exists(ContextInfo.order_file_path):
        return

    # Read orders
    try:
        with open(ContextInfo.order_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        log.error('Failed to read order file: %s', e)
        return

    orders = data.get('orders', [])
    if not orders:
        return

    # Execute each order
    for order in orders:
        order_id = order.get('order_id')
        if order_id in ContextInfo.submitted_orders:
            continue  # Already submitted

        _execute_order(ContextInfo, order)


def _execute_order(ContextInfo, order):
    """Execute a single order via passorder()."""
    order_id = order.get('order_id')
    side = order.get('side')
    symbol = order.get('symbol')
    target_qty = order.get('target_qty')
    price_limit = order.get('price_limit', 0)

    if not all([order_id, side, symbol, target_qty]):
        log.error('Missing required fields in order: %s', order)
        return

    # Convert symbol format: 600519 -> 600519.SH
    qmt_code = _to_qmt_code(symbol)

    # Determine order type
    if side == 'buy':
        op_type = STOCK_BUY
    elif side == 'sell':
        op_type = STOCK_SELL
    else:
        log.error('Unknown side: %s for order %s', side, order_id)
        return

    # Use limit price if available, otherwise latest price
    if price_limit and price_limit > 0:
        pr_type = FIX_PRICE
        price = price_limit
    else:
        pr_type = LATEST_PRICE
        price = 0

    log.info('Submitting order %s: %s %s qty=%d price=%.2f',
             order_id, side, qmt_code, target_qty, price)

    try:
        ContextInfo.passorder(
            op_type,        # opType: 23=buy, 24=sell
            0,              # orderType: 0=normal order
            ACCOUNT_ID,     # accountid
            qmt_code,       # orderCode: 600519.SH
            pr_type,        # prType: 11=limit, 5=latest
            price,          # modelprice
            target_qty      # volume in shares
        )
        ContextInfo.submitted_orders.add(order_id)
        log.info('Order %s submitted successfully', order_id)
    except Exception as e:
        log.error('Order %s failed: %s', order_id, e)
        _record_fill(ContextInfo, order_id, 'rejected', 0, 0, str(e))


def _record_fill(ContextInfo, order_id, status, fill_price, fill_qty, msg=''):
    """Record a fill result."""
    fill = {
        'order_id': order_id,
        'status': status,
        'fill_price': fill_price,
        'fill_qty': fill_qty,
        'fill_time': _now_str(),
        'message': msg,
    }
    ContextInfo.fill_results.append(fill)
    _write_fills(ContextInfo)


def _write_fills(ContextInfo):
    """Write fill results to file."""
    if not ContextInfo.fill_file_path:
        return

    fill_data = {
        'trade_date': ContextInfo.current_date,
        'submitted_at': _now_str(),
        'fills': ContextInfo.fill_results,
        'errors': [],
    }

    tmp_path = ContextInfo.fill_file_path + '.tmp'
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(fill_data, f, ensure_ascii=False, indent=2)
        # Atomic rename (os.rename overwrites on Windows)
        try:
            os.rename(tmp_path, ContextInfo.fill_file_path)
        except OSError:
            # Fallback: remove then rename
            if os.path.exists(ContextInfo.fill_file_path):
                os.remove(ContextInfo.fill_file_path)
            os.rename(tmp_path, ContextInfo.fill_file_path)
    except (IOError, OSError) as e:
        log.error('Failed to write fills: %s', e)


def order_callback(ContextInfo, order_info):
    """Called when order status changes."""
    log.info('Order callback: id=%s status=%s traded=%d',
             order_info.order_id, order_info.order_status,
             order_info.traded_volume)


def deal_callback(ContextInfo, deal_info):
    """Called when a trade executes."""
    log.info('Deal callback: %s %d@%.2f',
             deal_info.stock_code, deal_info.traded_volume,
             deal_info.traded_price)
    _record_fill(
        ContextInfo,
        deal_info.order_id,
        'filled',
        deal_info.traded_price,
        deal_info.traded_volume,
    )


def orderError_callback(ContextInfo, passorder_info, msg):
    """Called when an order fails."""
    log.error('Order error: %s', msg)
    # Extract order_id from passorder_info (may be dict or object)
    if isinstance(passorder_info, dict):
        order_id = passorder_info.get('order_id', 'unknown')
    else:
        order_id = getattr(passorder_info, 'order_id', 'unknown')
    _record_fill(ContextInfo, order_id, 'rejected', 0, 0, msg)


def stop(ContextInfo):
    """Strategy stopped."""
    log.info('QMT strategy stopped')
    _write_fills(ContextInfo)


def _to_qmt_code(symbol):
    """Convert bare symbol to QMT format: 600519 -> 600519.SH"""
    if '.' in symbol:
        return symbol
    if symbol.startswith(('6', '5', '9')):
        return '{}.SH'.format(symbol)
    if symbol.startswith(('0', '2', '3')):
        return '{}.SZ'.format(symbol)
    if symbol.startswith('8'):
        return '{}.BJ'.format(symbol)
    # Default to SZ for unknown
    return '{}.SZ'.format(symbol)


def _get_today(ContextInfo):
    """Get today's date from QMT framework."""
    try:
        timetag = ContextInfo.get_bar_timetag(ContextInfo.barpos)
        # timetag_to_datetime is injected by QMT framework
        dt_str = timetag_to_datetime(timetag, '%Y-%m-%d')
        return dt_str
    except Exception:
        # Fallback to system date (may differ on weekends/holidays)
        import datetime
        return datetime.date.today().isoformat()


def _now_str():
    """Current time as ISO string."""
    import datetime
    return datetime.datetime.now().isoformat()
