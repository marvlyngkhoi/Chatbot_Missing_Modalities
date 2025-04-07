from trl import SFTConfig
from peft import LoraConfig
# gemma_tr_config = SFTConfig(
    
# )

def get_peft_configs(
    alpha=16,r_val=8,dropout=0.2,
    target_modules=["q_proj","v_proj"],
    task_type="CAUSAL_LM"
):
    
    return LoraConfig(
        lora_alpha=alpha,
        r=r_val,
        target_modules=target_modules,
        task_type=task_type
    )
    

def get_configs(output_dir,
                epochs=10,
                train_batch=8,
                eval_batch=8,
                grad_accumulation=8,
                lr = 2e-4,
                eval_strategy="epoch",
                optim='adamw_torch_fused',
                metric_for_best_model='eval_loss',
                max_grad_norm=0.3,
                logging_steps=100,
                gradient_chekpointig=True,
                ds_field="",
                ds_kwargs={"skip_prepare_dataset": True},

                ):
    training_arg = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_eval_batch_size=eval_batch,
        per_device_train_batch_size=train_batch,
        learning_rate=lr,
        optim=optim,
        logging_steps=logging_steps,
        gradient_checkpointing=gradient_chekpointig,
        eval_strategy=eval_strategy,
        max_grad_norm=max_grad_norm,
        dataset_text_field=ds_field,
        dataset_kwargs=ds_kwargs,
    )

    return training_arg


def get_configs_llama(output_dir,
                epochs=10,
                train_batch=8,
                eval_batch=8,
                grad_accumulation=8,
                lr = 2e-4,
                eval_strategy="epoch",
                optim='adamw_torch_fused',
                metric_for_best_model='eval_loss',
                max_grad_norm=0.3,
                logging_steps=100,
                gradient_chekpointig=True,
                ds_field="",
                ds_kwargs={"skip_prepare_dataset": True},

                ):
    training_arg = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_eval_batch_size=eval_batch,
        per_device_train_batch_size=train_batch,
        learning_rate=lr,
        optim=optim,
        logging_steps=logging_steps,
        gradient_checkpointing=gradient_chekpointig,
        eval_strategy=eval_strategy,
        max_grad_norm=max_grad_norm,
        dataset_text_field=ds_field,
        dataset_kwargs=ds_kwargs,
    )

    return training_arg




