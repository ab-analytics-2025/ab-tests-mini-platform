type Props = {
  title: string;
  data: any;
};

export default function JsonBlock({ title, data }: Props) {
  return (
    <div className="card">
      <h2>{title}</h2>
      <pre className="code">{JSON.stringify(data, null, 2)}</pre>
    </div>
  );
}
