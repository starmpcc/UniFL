import torch


def communication(args, server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params
        if args.algorithm == "fedbn" or args.algorithm == "fedpxn":
            for key in server_model.state_dict().keys():
                if "norm" not in key:
                    temp = torch.zeros_like(
                        server_model.state_dict()[key], dtype=torch.float32
                    )

                    for client_idx in range(len(args.src_data)):
                        temp += (
                            client_weights[client_idx]
                            * models[client_idx].state_dict()[key]
                        )

                    server_model.state_dict()[key].data.copy_(temp)

                    for client_idx in range(len(args.src_data)):
                        models[client_idx].state_dict()[key].data.copy_(
                            server_model.state_dict()[key]
                        )

        else:  # fedavg, fedprox
            for key in server_model.state_dict().keys():
                if "num_batches_tracked" in key:
                    server_model.state_dict()[key].data.copy_(
                        models[0].state_dict()[key]
                    )
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])

                    for client_idx in range(len(args.src_data)):
                        temp += (
                            client_weights[client_idx]
                            * models[client_idx].state_dict()[key]
                        )

                    server_model.state_dict()[key].data.copy_(temp)

                    for client_idx in range(len(args.src_data)):
                        models[client_idx].state_dict()[key].data.copy_(
                            server_model.state_dict()[key]
                        )

    return server_model, models
