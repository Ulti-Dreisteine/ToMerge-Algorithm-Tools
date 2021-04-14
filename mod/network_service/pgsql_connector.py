import psycopg2


class PostgresqlConnector(object):
	"""
    PGSQL连接器
    """
	
	def __init__(self, config):
		self.conn = psycopg2.connect(**config)
		self.cursor = self.conn.cursor()
	
	def __commit__(self):
		self.conn.commit()
	
	def __rollback__(self):
		self.conn.rollback()
	
	def execute(self, sql: str):
		"""
        执行SQL语句
        
        Example:
        ------------------------------------------------------------
        pc = PostgresqlConnector(config_params)
		response = pc.execute("SELECT * FROM t_pollution_forecast WHERE f_area_code = '{}'".format(area_code)).fetchall()
		pc.close()
		------------------------------------------------------------
        """
		try:
			self.cursor.execute(sql)
		except psycopg2.InternalError as e:
			if e.pgerror == 'ERROR:  current transaction is aborted, commands ignored until end of transaction block\n':
				self.__rollback__()
				return self.execute(sql)
			raise e
		else:
			self.__commit__()
			return self.cursor
	
	def close(self):
		self.cursor.close()
		self.conn.close()
