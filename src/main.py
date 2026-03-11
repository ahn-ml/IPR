from .env import DEBUG


if DEBUG:
    from jaxtyping import install_import_hook
    import torch

    torch.set_printoptions(
        threshold=8,
        edgeitems=2
    )

    # Configure beartype and jaxtyping.
    with install_import_hook(
        ("src",),
        ("beartype", "beartype"),
    ):
        from src._main import main
else:
    from src._main import main


if __name__ == "__main__":
    main()
