############################ GeoLayoutLMWrapper tracing #############################
import torch
from transformers import BertTokenizer
from PIL import Image
import torchvision.transforms.functional as F

class GeoLayoutLMWrapper(torch.nn.Module):
    def __init__(self, model):
        super(GeoLayoutLMWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, bbox, image, attention_mask, el_label_seq_mask,
                are_box_first_tokens, size_raw, bbox_4p_normalized, 
                first_token_idxes, block_mask, line_rank_id, line_rank_inner_id,
                bio_labels, el_labels_seq, el_labels_blk, el_label_blk_mask):
        device = next(self.model.parameters()).device
        input_data = {
            'input_ids': input_ids, 'bbox': bbox, 'image': image, 
            'attention_mask': attention_mask, 'el_label_seq_mask': el_label_seq_mask, 
            'are_box_first_tokens': are_box_first_tokens, 'size_raw': size_raw, 
            'bbox_4p_normalized': bbox_4p_normalized, 'first_token_idxes': first_token_idxes,
            'block_mask': block_mask, 'line_rank_id': line_rank_id, 
            'line_rank_inner_id': line_rank_inner_id, 'bio_labels': bio_labels, 
            'el_labels_seq': el_labels_seq, 'el_labels_blk': el_labels_blk, 
            'el_label_blk_mask': el_label_blk_mask
        }
        ##########  approch 1 not working #################################           
        # Ensure all inputs are tensors and have a batch dimension
        # for k, v in input_data.items():
        #     if not isinstance(v, torch.Tensor):
        #         input_data[k] = torch.tensor(v)
        #     if input_data[k].dim() == 1:
        #         input_data[k] = input_data[k].unsqueeze(0)
        ###########################################           
        for k in input_data.keys():
            if isinstance(input_data[k], torch.Tensor):
                input_data[k] = input_data[k].to(device)
                input_data[k] = input_data[k].unsqueeze(0)
        with torch.no_grad():
            head_outputs, loss_dict = self.model(input_data)
        print(head_outputs["logits4labeling"])
        print(f"Type: {type(head_outputs['logits4labeling'])}, Shape: {head_outputs['logits4labeling'].shape}")
        return head_outputs["logits4labeling"]
    
    
    
#calling
from transformers import BertTokenizer
 
def predict(net, image, json_obj, class_names_path, backbone_type='geolayoutlm'):
    net.eval()
    # tokenizer = net.tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_TOKENIZER_PATH, do_lower_case=True)
    input_data = getitem_geo(image, json_obj, tokenizer, backbone_type)
    ### print shape ########
    # for k in input_data.keys():
    #     if isinstance(input_data[k], torch.Tensor):
    #         print(f"Key: {k}, Shape: {input_data[k].shape}, Type: {input_data[k].dtype}")'''
    ###################################################
    ################ tracing ###################################
    model_wrapper = GeoLayoutLMWrapper(net)
 
    traced_model = torch.jit.trace(
        model_wrapper,
        example_inputs=(
            input_data['input_ids'],
            input_data['bbox'],
            input_data['image'],
            input_data['attention_mask'],
            input_data['el_label_seq_mask'],
            input_data['are_box_first_tokens'],
            input_data['size_raw'],
            input_data['bbox_4p_normalized'],
            input_data['first_token_idxes'],
            input_data['block_mask'],
            input_data['line_rank_id'],
            input_data['line_rank_inner_id'],
            input_data['bio_labels'],
            input_data['el_labels_seq'],
            input_data['el_labels_blk'],
            input_data['el_label_blk_mask']
        ),
        check_trace=False,
        strict=False
    )
    torch.jit.save(traced_model, "traced_geolayoutlm_model.pt")
    print('Tracing completed and model saved successfully.')