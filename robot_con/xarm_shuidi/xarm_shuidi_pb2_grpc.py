# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import robot_con.xarm_shuidi.xarm_shuidi_pb2 as xarm__shuidi__pb2


class XArmShuidiStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.arm_move_jspace_path = channel.unary_unary(
                '/XArmShuidi/arm_move_jspace_path',
                request_serializer=xarm__shuidi__pb2.Path.SerializeToString,
                response_deserializer=xarm__shuidi__pb2.Status.FromString,
                )
        self.arm_get_jnt_values = channel.unary_unary(
                '/XArmShuidi/arm_get_jnt_values',
                request_serializer=xarm__shuidi__pb2.Empty.SerializeToString,
                response_deserializer=xarm__shuidi__pb2.JntValues.FromString,
                )
        self.arm_jaw_to = channel.unary_unary(
                '/XArmShuidi/arm_jaw_to',
                request_serializer=xarm__shuidi__pb2.GripperStatus.SerializeToString,
                response_deserializer=xarm__shuidi__pb2.Status.FromString,
                )
        self.arm_get_gripper_status = channel.unary_unary(
                '/XArmShuidi/arm_get_gripper_status',
                request_serializer=xarm__shuidi__pb2.Empty.SerializeToString,
                response_deserializer=xarm__shuidi__pb2.GripperStatus.FromString,
                )
        self.agv_move = channel.unary_unary(
                '/XArmShuidi/agv_move',
                request_serializer=xarm__shuidi__pb2.Speed.SerializeToString,
                response_deserializer=xarm__shuidi__pb2.Status.FromString,
                )


class XArmShuidiServicer(object):
    """Missing associated documentation comment in .proto file."""

    def arm_move_jspace_path(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def arm_get_jnt_values(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def arm_jaw_to(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def arm_get_gripper_status(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def agv_move(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_XArmShuidiServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'arm_move_jspace_path': grpc.unary_unary_rpc_method_handler(
                    servicer.arm_move_jspace_path,
                    request_deserializer=xarm__shuidi__pb2.Path.FromString,
                    response_serializer=xarm__shuidi__pb2.Status.SerializeToString,
            ),
            'arm_get_jnt_values': grpc.unary_unary_rpc_method_handler(
                    servicer.arm_get_jnt_values,
                    request_deserializer=xarm__shuidi__pb2.Empty.FromString,
                    response_serializer=xarm__shuidi__pb2.JntValues.SerializeToString,
            ),
            'arm_jaw_to': grpc.unary_unary_rpc_method_handler(
                    servicer.arm_jaw_to,
                    request_deserializer=xarm__shuidi__pb2.GripperStatus.FromString,
                    response_serializer=xarm__shuidi__pb2.Status.SerializeToString,
            ),
            'arm_get_gripper_status': grpc.unary_unary_rpc_method_handler(
                    servicer.arm_get_gripper_status,
                    request_deserializer=xarm__shuidi__pb2.Empty.FromString,
                    response_serializer=xarm__shuidi__pb2.GripperStatus.SerializeToString,
            ),
            'agv_move': grpc.unary_unary_rpc_method_handler(
                    servicer.agv_move,
                    request_deserializer=xarm__shuidi__pb2.Speed.FromString,
                    response_serializer=xarm__shuidi__pb2.Status.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'XArmShuidi', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class XArmShuidi(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def arm_move_jspace_path(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/XArmShuidi/arm_move_jspace_path',
            xarm__shuidi__pb2.Path.SerializeToString,
            xarm__shuidi__pb2.Status.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def arm_get_jnt_values(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/XArmShuidi/arm_get_jnt_values',
            xarm__shuidi__pb2.Empty.SerializeToString,
            xarm__shuidi__pb2.JntValues.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def arm_jaw_to(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/XArmShuidi/arm_jaw_to',
            xarm__shuidi__pb2.GripperStatus.SerializeToString,
            xarm__shuidi__pb2.Status.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def arm_get_gripper_status(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/XArmShuidi/arm_get_gripper_status',
            xarm__shuidi__pb2.Empty.SerializeToString,
            xarm__shuidi__pb2.GripperStatus.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def agv_move(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/XArmShuidi/agv_move',
            xarm__shuidi__pb2.Speed.SerializeToString,
            xarm__shuidi__pb2.Status.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
