-- MySQL dump 10.13  Distrib 9.0.1, for Linux (x86_64)
--
-- Host: localhost    Database: federate_codigla
-- ------------------------------------------------------
-- Server version	9.0.1

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `AdversarialRun`
--

DROP TABLE IF EXISTS `AdversarialRun`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `AdversarialRun` (
  `adv_id` int NOT NULL AUTO_INCREMENT,
  `result_id` int NOT NULL,
  `attack_type` varchar(10) NOT NULL,
  `epsilon` float NOT NULL,
  `eps_step` float DEFAULT NULL,
  `max_iter` int NOT NULL DEFAULT '10',
  `num_random_init` int NOT NULL DEFAULT '1',
  `n_samples` int DEFAULT NULL,
  `clean_bal_acc` float NOT NULL,
  `adv_bal_acc` float NOT NULL,
  `clean_precision` float NOT NULL,
  `adv_precision` float NOT NULL,
  `clean_recall` float NOT NULL,
  `adv_recall` float NOT NULL,
  PRIMARY KEY (`adv_id`),
  KEY `fk_adv_result` (`result_id`),
  CONSTRAINT `fk_adv_result` FOREIGN KEY (`result_id`) REFERENCES `TrainingResult` (`result_id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=31 DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `Dataset`
--

DROP TABLE IF EXISTS `Dataset`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Dataset` (
  `dataset_id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(100) NOT NULL,
  `path` varchar(500) NOT NULL,
  `total_samples` int NOT NULL,
  `samples_class0` int NOT NULL,
  `samples_class1` int NOT NULL,
  `description` longtext,
  `mock_image_dim1` int DEFAULT '1',
  `mock_image_dim2` int DEFAULT '3',
  `mock_image_dim3` int DEFAULT '256',
  `mock_image_dim4` varchar(100) DEFAULT '256',
  `mock_image_min` int DEFAULT '0',
  `mock_image_max` int DEFAULT '255',
  `mock_image_samples0` int DEFAULT '10',
  `mock_image_samples1` int DEFAULT '10',
  `mock_path` varchar(500) DEFAULT './mock_data',
  PRIMARY KEY (`dataset_id`),
  UNIQUE KEY `uq_dataset_name` (`name`)
) ENGINE=InnoDB AUTO_INCREMENT=58 DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `Experiment`
--

DROP TABLE IF EXISTS `Experiment`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Experiment` (
  `experiment_id` int NOT NULL AUTO_INCREMENT,
  `description` varchar(300) DEFAULT NULL,
  `lr` float NOT NULL,
  `batch_size` int NOT NULL,
  `epochs_max` int NOT NULL,
  `patience` int NOT NULL,
  `eve_model_path` varchar(100) DEFAULT NULL,
  `servers_base_port` int DEFAULT '20000',
  PRIMARY KEY (`experiment_id`)
) ENGINE=InnoDB AUTO_INCREMENT=84 DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `ExperimentSplit`
--

DROP TABLE IF EXISTS `ExperimentSplit`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `ExperimentSplit` (
  `es_id` int NOT NULL AUTO_INCREMENT,
  `experiment_id` int NOT NULL,
  `split_id` int NOT NULL,
  `result_id` int DEFAULT NULL,
  PRIMARY KEY (`es_id`),
  UNIQUE KEY `uq_exp_split` (`experiment_id`,`split_id`),
  KEY `fk_es_split` (`split_id`),
  KEY `fk_es_result` (`result_id`),
  CONSTRAINT `fk_es_experiment` FOREIGN KEY (`experiment_id`) REFERENCES `Experiment` (`experiment_id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_es_result` FOREIGN KEY (`result_id`) REFERENCES `TrainingResult` (`result_id`) ON DELETE SET NULL ON UPDATE CASCADE,
  CONSTRAINT `fk_es_split` FOREIGN KEY (`split_id`) REFERENCES `Split` (`split_id`) ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=155 DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `MembershipInferenceRun`
--

DROP TABLE IF EXISTS `MembershipInferenceRun`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `MembershipInferenceRun` (
  `mia_id` int NOT NULL AUTO_INCREMENT,
  `result_id` int NOT NULL,
  `attack_variant` varchar(30) NOT NULL,
  `n_train_samples` int NOT NULL,
  `n_test_samples` int NOT NULL,
  `mia_accuracy` float NOT NULL,
  `mia_precision` float NOT NULL,
  `mia_recall` float NOT NULL,
  PRIMARY KEY (`mia_id`),
  KEY `fk_mia_result` (`result_id`),
  CONSTRAINT `fk_mia_result` FOREIGN KEY (`result_id`) REFERENCES `TrainingResult` (`result_id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `PoisoningRun`
--

DROP TABLE IF EXISTS `PoisoningRun`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `PoisoningRun` (
  `poison_id` int NOT NULL AUTO_INCREMENT,
  `result_id` int NOT NULL,
  `trigger_type` varchar(20) NOT NULL,
  `trigger_size` int DEFAULT NULL,
  `trigger_position` varchar(20) DEFAULT NULL,
  `percent_poison` float NOT NULL,
  `n_poisoned` int NOT NULL,
  `source_class` int NOT NULL,
  `target_class` int NOT NULL,
  `clean_bal_acc` float NOT NULL,
  `clean_precision` float NOT NULL,
  `clean_recall` float NOT NULL,
  `attack_success_rate` float NOT NULL,
  `ac_precision` float NOT NULL,
  `ac_recall` float NOT NULL,
  `ac_f1` float NOT NULL,
  PRIMARY KEY (`poison_id`),
  KEY `fk_poison_result` (`result_id`),
  CONSTRAINT `fk_poison_result` FOREIGN KEY (`result_id`) REFERENCES `TrainingResult` (`result_id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `Server`
--

DROP TABLE IF EXISTS `Server`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Server` (
  `server_id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(100) CHARACTER SET utf8mb3 COLLATE utf8mb3_general_ci NOT NULL,
  `owner_client_password` varchar(100) CHARACTER SET utf8mb3 COLLATE utf8mb3_general_ci NOT NULL DEFAULT 'changethis',
  `owner_client_email` varchar(100) CHARACTER SET utf8mb3 COLLATE utf8mb3_general_ci NOT NULL DEFAULT 'info@openmined.org',
  `owner_client_name` varchar(100) CHARACTER SET utf8mb3 COLLATE utf8mb3_general_ci DEFAULT 'Alice',
  `owner_client_institution` varchar(100) CHARACTER SET utf8mb3 COLLATE utf8mb3_general_ci DEFAULT 'Alice Place',
  PRIMARY KEY (`server_id`),
  UNIQUE KEY `Server_UNIQUE` (`name`)
) ENGINE=InnoDB AUTO_INCREMENT=147 DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `Split`
--

DROP TABLE IF EXISTS `Split`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Split` (
  `split_id` int NOT NULL AUTO_INCREMENT,
  `dataset_id` int NOT NULL,
  `seed` int NOT NULL DEFAULT '42',
  `train_ratio` float NOT NULL DEFAULT '0.7',
  `val_ratio` float NOT NULL DEFAULT '0.1',
  `n_train` int NOT NULL DEFAULT '0',
  `n_val` int NOT NULL DEFAULT '0',
  `n_test` int NOT NULL DEFAULT '0',
  `server_id` int NOT NULL,
  `model_path` varchar(100) DEFAULT './model',
  `epochs` int DEFAULT '1',
  PRIMARY KEY (`split_id`),
  KEY `fk_split_dataset` (`dataset_id`),
  KEY `Split_Server_FK` (`server_id`),
  CONSTRAINT `fk_split_dataset` FOREIGN KEY (`dataset_id`) REFERENCES `Dataset` (`dataset_id`) ON DELETE RESTRICT ON UPDATE CASCADE,
  CONSTRAINT `Split_Server_FK` FOREIGN KEY (`server_id`) REFERENCES `Server` (`server_id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE=InnoDB AUTO_INCREMENT=158 DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `TrainingResult`
--

DROP TABLE IF EXISTS `TrainingResult`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `TrainingResult` (
  `result_id` int NOT NULL AUTO_INCREMENT,
  `model_path` varchar(500) NOT NULL,
  `best_epoch` int NOT NULL,
  `best_val_bal_acc` float NOT NULL,
  `test_bal_acc` float NOT NULL,
  `test_precision` float NOT NULL,
  `test_recall` float NOT NULL,
  PRIMARY KEY (`result_id`)
) ENGINE=InnoDB AUTO_INCREMENT=13 DEFAULT CHARSET=utf8mb3;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping routines for database 'federate_codigla'
--
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2026-05-20 11:14:24
