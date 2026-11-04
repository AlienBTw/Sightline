import { Alert } from 'react-native';
import { C, CAMERA_STATE, S } from '../lib/theme';
import type { Camera } from '../lib/types';
import { Sheet } from './Sheet';
import { Button, Txt } from './ui';

export function CameraOptionsSheet({
  cam,
  canControl,
  canDelete,
  busy,
  onClose,
  onToggle,
  onDeleted,
}: {
  cam: Camera | null;
  canControl: boolean;
  canDelete: boolean;
  busy: boolean;
  onClose: () => void;
  onToggle: (cam: Camera) => void | Promise<void>;
  onDeleted: (cam: Camera) => void | Promise<void>;
}) {
  if (!cam) return null;
  const target = cam;

  const st = CAMERA_STATE[target.status.state] ?? { label: target.status.state, color: C.muted };
  const active = !['stopped', 'error', 'finished'].includes(target.status.state);

  function confirmDelete() {
    Alert.alert('Delete camera?', `“${target.name}” will be removed. Its alerts are kept.`, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: () => {
          onClose();
          void onDeleted(target);
        },
      },
    ]);
  }

  return (
    <Sheet visible title={target.name} onClose={onClose} dismissable={!busy}>
      <Txt variant="muted">{st.label}</Txt>
      {canControl ? (
        <Button
          title={active ? 'Stop' : 'Start'}
          icon={active ? 'stop' : 'play'}
          kind={active ? 'secondary' : 'primary'}
          loading={busy}
          onPress={() => void onToggle(target)}
        />
      ) : null}
      {canDelete ? (
        <Button title="Delete" icon="trash-outline" kind="danger" disabled={busy} onPress={confirmDelete} />
      ) : null}
      {!canControl && !canDelete ? (
        <Txt variant="muted" style={{ marginTop: S.sm }}>
          An admin can start, stop, or remove this camera.
        </Txt>
      ) : null}
    </Sheet>
  );
}
