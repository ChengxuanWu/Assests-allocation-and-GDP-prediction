from self_func import connect_Mysql, create_tables

db = connect_Mysql()
#build up table for storing predictors
sqlsyntax = '''CREATE TABLE IF NOT EXISTS `predictors` (
`index` CHAR NOT NULL,
`exportation` INT NOT NULL,
`importation` INT NOT NULL,
`m1a` INT NOT NULL,
`m1b` INT NOT NULL,
`m2` INT NOT NULL,
`stock` INT NOT NULL,
`salary` INT NOT NULL,
`worktime` FLOAT NOT NULL,
`workforce` INT NOT NULL,
`preliminary` FLOAT NOT NULL,
`industrail_index` FLOAT NOT NULL,
`torism` INT NOT NULL,
`order` INT NOT NULL,
`cpi` FLOAT NOT NULL);'''
create_tables(db, sqlsyntax)
 
#create table-GDP(target)
sqlsyntax = '''CREATE TABLE IF NOT EXISTS `targets` (
`id` INT PRIMARY KEY AUTO_INCREMENT,
`GDP_growth_rate(%)` float NOT NULL,
`GDP_chian` INT NOT NULL,
`index_label` CHAR NOT NULL);'''
create_tables(db, sqlsyntax)  
 
#setup for time and quartr for stocks as foreign key
sqlsyntax = '''CREATE TABLE IF NOT EXISTS `stocks_time` (
`id` INT PRIMARY KEY AUTO_INCREMENT,
`year` INT NOT NULL,
`mon` INT NOT NULL,
`day` INT NOT NULL,
`quarter` INT NOT NULL);'''
create_tables(db, sqlsyntax)

#assets table for etfs and stocks data
sqlsyntax = '''CREATE TABLE IF NOT EXISTS `stocks` (
`id` INT PRIMARY KEY AUTO_INCREMENT,
`0050_price(NT)` INT NOT NULL,
`00679B_price(NT)` INT NOT NULL,
`00713_price(NT)` INT NOT NULL,
FOREIGN KEY(id) REFERENCES stocks_time(id));'''
create_tables(db, sqlsyntax)

