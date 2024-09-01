# -*- coding: utf-8 -*-

from OpenApi import *
from sqlalchemy import create_engine
import pymysql

pymysql.install_as_MySQLdb()


class Collector():
    def __init__(self):
        self.api = Openapi.Openapi()

    def db_setting(self, db_name):
        db_name = db_name

        # mysql 데이터베이스랑 연동하는 방식.
        # bot: 계정명
        # 1234!@#$ : 본인 데이터베이스의 비밀번호를 입력
        # localhost : 자신의 PC에 데이터베이스를 구축한 경우 (만약 다른 PC에 데이터베이스를 구축한 경우는 해당 PC의 IP를 기재)
        # 3306 : mysql 접속 기본 포트 번호 (만약 포트 번호를 변경 하신 분은 그에 맞게 설정 해주셔야 합니다.)
        self.engine_bot = create_engine("mysql+mysqldb://bot:" + "1234!@#$" + "@localhost:3306/" + db_name,
                                        encoding='utf-8')

        # 데이터베이스에 실행 할 쿼리
        sql = "select * from bot_test1.class1;"

        # 위의 sql 문을 데이터베이스에 실행한 결과를 rows라는 변수에 담는다.
        rows = self.engine_bot.execute(sql).fetchall()
        print(rows)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    collector = Collector()
    c.db_setting()