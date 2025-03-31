class STGCNError(Exception):
    """STGCN 基礎異常類"""
    pass

class DataError(STGCNError):
    """數據相關錯誤"""
    pass

class ModelError(STGCNError):
    """模型相關錯誤"""
    pass

class ConfigError(STGCNError):
    """配置相關錯誤"""
    pass

class TrainingError(STGCNError):
    """訓練相關錯誤"""
    pass

class ValidationError(STGCNError):
    """驗證相關錯誤"""
    pass

class GPUError(STGCNError):
    """GPU 相關錯誤"""
    pass

class CheckpointError(STGCNError):
    """檢查點相關錯誤"""
    pass

class DataFormatError(DataError):
    """數據格式錯誤"""
    pass

class DataNotFoundError(DataError):
    """數據文件未找到錯誤"""
    pass

class ModelNotInitializedError(ModelError):
    """模型未初始化錯誤"""
    pass

class InvalidConfigError(ConfigError):
    """無效配置錯誤"""
    pass

class TrainingInterruptedError(TrainingError):
    """訓練中斷錯誤"""
    pass

class ValidationFailedError(ValidationError):
    """驗證失敗錯誤"""
    pass

class GPUNotAvailableError(GPUError):
    """GPU 不可用錯誤"""
    pass

class CheckpointNotFoundError(CheckpointError):
    """檢查點未找到錯誤"""
    pass 