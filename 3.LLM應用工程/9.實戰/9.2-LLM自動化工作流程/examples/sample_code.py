"""示例代碼 - 用於測試代碼審查功能"""


def calculate_total(items):
    """計算商品總價"""
    total = 0
    for item in items:
        total = total + item['price']
    return total


def find_max(numbers):
    """找到列表中的最大值"""
    if len(numbers) == 0:
        return None
    max_val = numbers[0]
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val


def process_user_data(data):
    """處理用戶數據"""
    try:
        result = []
        for item in data:
            if item['age'] > 18:
                result.append(item)
        return result
    except (KeyError, TypeError):
        return []


class UserManager:
    """用戶管理器"""

    def __init__(self, users=[]):
        self.users = users

    def add_user(self, user):
        self.users.append(user)

    def get_user(self, user_id):
        for user in self.users:
            if user['id'] == user_id:
                return user
        return None

    def delete_user(self, user_id):
        for i in range(len(self.users)):
            if self.users[i]['id'] == user_id:
                del self.users[i]
                return True
        return False


def complex_function(a, b, c, d, e):
    """一個複雜的函數"""
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        return a + b + c + d + e
                    else:
                        return a + b + c + d
                else:
                    return a + b + c
            else:
                return a + b
        else:
            return a
    else:
        return 0


# 全局變量
GLOBALVAR = "test"


def use_global():
    """使用全局變量"""
    print(GLOBALVAR)
    return GLOBALVAR
